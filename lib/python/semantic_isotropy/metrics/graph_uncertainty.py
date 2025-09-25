from semantic_isotropy.llm.api import chat_api
from semantic_isotropy.llm.utils import estimate_tokens
from semantic_isotropy.pipeline.utils import init_logger
from semantic_isotropy.llm.utils import estimate_tokens, TokenRateLimiter

import networkx as nx

from concurrent.futures import ThreadPoolExecutor, as_completed


## Implementation taken from (with minor modifications): https://github.com/jiangjmj/Graph-based-Uncertainty/tree/main
# @misc{jiang2024graphbaseduncertaintymetricslongform,
#       title={Graph-based Uncertainty Metrics for Long-form Language Model Outputs},
#       author={Mingjian Jiang and Yangjun Ruan and Prasanna Sattigeri and Salim Roukos and Tatsunori Hashimoto},
#       year={2024},
#       eprint={2410.20783},
#       archivePrefix={arXiv},
#       primaryClass={cs.CL},
#       url={https://arxiv.org/abs/2410.20783},
# }

logger = init_logger(__name__, 'INFO')

CLAIM_UNION_PROMPT =   """Given two lists titled "Original Claim List" and "New Claim List", your task is to integrate information from the "New Claim List" into the "Original Claim List". Please follow these detailed steps to ensure accuracy and clarity in the process:\n\nTask 1. **Verification Process:**  Your goal is to go through each statement in the "New Claim List" one by one, and determine if it is fully entailed or mentioned by any statement in the "Original Claim List." \n\nTask 2. **Compilation of Non-Entailed Claims:** Generate a list of statements from the "New Claim List" that are not already covered or implied by the "Original Claim List." For each new or unique claim that does not have an equivalent in the original list, format your output by starting each line with a dash ('-').\n\n**Original Claim List:**\n{original_claim_list}\n\n**New Claim List:**\n{new_claim_list}\n\nBegin with the Verification Process to assess each claim's relevance and uniqueness, followed by the Compilation of Non-Entailed Claims to clearly list any new insights that the "New Claim List" provides.
The "New Claim List" must be provided between <new_claims></new_claims> tags."""

RESPONSE_ENTAILMENT_PROMPT = """Follow the instructions carefully. Given a response and a claim, determine if the response entails the claim. If it does, return "T" for True. If it does not, return "F" for False. Return only the letter, no other text.\n\n**Response:**\n{response}\n\n**Claim:**\n{claim}"""

def metrics_for_bipartite_graph(
    G: nx.Graph,
    side: int = 0,
    weight: Optional[str] = None,
    pagerank_d: float = 0.85,
    ev_method: str = "numpy",     # "power" or "numpy"
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """
    Compute Degree, Betweenness, Eigenvector, PageRank, and Closeness centralities
    for the nodes on one side of a bipartite graph.

    Parameters
    ----------
    G : nx.Graph
        bipartite graph (undirected or directed). Nodes should have attribute `bipartite` in {0,1}.
    side : int
        Which bipartite set to report metrics for (0 or 1). Default 0.
    weight : str | None
        Edge attribute to treat as weight (e.g., "weight"). If None, unweighted.
    pagerank_d : float
        Damping factor 'd' used by PageRank (NetworkX `alpha`).
    ev_method : {"power","numpy"}
        Backend for eigenvector centrality. "numpy" uses exact eigen-solve;
        "power" uses iterative power method (respects `max_iter`/`tol`).
    max_iter : int
        Max iterations for iterative methods.
    tol : float
        Convergence tolerance for iterative methods.

    Returns
    -------
    pd.DataFrame
        Index: nodes with `bipartite==side`.
        Columns: ["degree", "betweenness", "eigenvector", "pagerank", "closeness"].
    """
    # --- identify the requested side ---
    side_nodes: Iterable[Hashable] = [n for n, d in G.nodes(data=True) if d.get("bipartite") == side]
    if not side_nodes:
        raise ValueError(f"No nodes found with bipartite={side}. Make sure nodes carry the 'bipartite' attribute.")

    # --- Degree (number of incident edges; weighted degree if weight provided) ---
    # For weighted degree, use sum of weights; otherwise plain degree count.
    if weight is None:
        deg: Dict[Hashable, float] = dict(G.degree())
    else:
        deg = dict(G.degree(weight=weight))

    # --- Betweenness centrality (fraction of shortest paths through v) ---
    # Works for undirected or directed graphs; normalized to [0,1].
    bet = nx.betweenness_centrality(G, normalized=True, weight=weight)

    # --- Eigenvector centrality (importance via neighbors' importance) ---
    if ev_method == "power":
        eig = nx.eigenvector_centrality(G, max_iter=max_iter, tol=tol, weight=weight)
    else:
        # numpy method is robust and fast for moderate graphs
        eig = nx.eigenvector_centrality_numpy(G, weight=weight)

    # --- PageRank (stationary distribution with damping) ---
    # For undirected graphs, NetworkX treats edges as bidirectional.
    pr = nx.pagerank(G, alpha=pagerank_d, max_iter=max_iter, tol=tol, weight=weight)

    # --- Closeness (reciprocal of avg shortest-path distance to all nodes) ---
    # Use the improved Wasserman–Faust normalization for disconnected graphs.
    clo = nx.closeness_centrality(G, distance=weight, wf_improved=True)

    # --- Assemble only the requested side ---
    res = {
            "degree": {n: deg.get(n, 0.0) for n in side_nodes},
            "betweenness": {n: bet.get(n, 0.0) for n in side_nodes},
            "eigenvector": {n: eig.get(n, 0.0) for n in side_nodes},
            "pagerank": {n: pr.get(n, 0.0) for n in side_nodes},
            "closeness": {n: clo.get(n, 0.0) for n in side_nodes},
    }
    return res

def graph_uncertainty(responses, factscore_data, response_idx, subset_idx, total_responses, rate_limiter, api_key, max_workers=30):
    fs_slice = factscore_data[response_idx:response_idx + total_responses]
    fs_slice_subset = [fs_slice[i] for i in subset_idx]

    original_claims_list = [fs_claim['atom'] for fs_claim in fs_slice_subset[0]]

    for fs_item in fs_slice_subset[1:]:
        new_claims_list = [fs_claim['atom'] for fs_claim in fs_item]
        prompt = CLAIM_UNION_PROMPT.format(original_claim_list=original_claims_list, new_claim_list=new_claims_list)
        rate_limiter[0].add_tokens(estimate_tokens(prompt))
        rate_limiter[1].add_tokens(1)
        try:
            tres = chat_api(prompt, api='openai', model='gpt-4.1-mini', api_key=api_key)
            unverified_claims_list = [claim.strip(' ').lstrip('-') for claim in tres['response'].split('<new_claims>')[1].split('</new_claims>')[0].strip().split('\n')]
        except Exception as e:
            tres = chat_api(test_prompt, api='gemini', model='gemini-2.5-flash', temperature=0.0)
            unverified_claims_list = [claim.strip(' ').lstrip('-') for claim in tres['response'].split('<new_claims>')[1].split('</new_claims>')[0].strip().split('\n')]
        original_claims_list.extend(unverified_claims_list)

    claim_entailment_map = {cidx: [] for cidx in range(len(original_claims_list))}
    print("Collapsed Claims Successfully")

    for ridx, response in enumerate(responses):
        resp_text = response['response']
        entailment_prompt_list = [RESPONSE_ENTAILMENT_PROMPT.format(response=resp_text, claim=claim) for claim in original_claims_list]

        def process_prompt_partial(rate_limiter, result_idx):
            def process_prompt(prompt, claim_idx):
                rate_limiter[0].add_tokens(estimate_tokens(prompt))
                rate_limiter[1].add_tokens(1)
                result = chat_api(prompt, api='openai', model='gpt-4.1-mini', api_key=api_key)['response']
                assert result.lower() in ['yes', 'no'], f"Invalid response for entailment_prompt claim_idx {claim_idx} = result_idx {result_idx}"
                return result.lower() == 'yes', claim_idx

            return process_prompt

        judge_entailment_partial = process_prompt_partial(rate_limiter, ridx)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_row = {
                executor.submit(judge_entailment_partial, entailment_prompt, idx): (entailment_prompt, idx)
                for idx, entailment_prompt in enumerate(entailment_prompt_list)
            }

            # Collect results as they complete
            for future in as_completed(future_to_row):
                try:
                    result = future.result()
                    if result is not None:
                        result_indicator, claim_idx = result
                        if result_indicator:
                            claim_entailment_map[claim_idx].append(ridx)
                except Exception as e:
                    print(f"Error processing response:{ridx}: {str(e)}")

        print(f"Entailed Claims Successfully for response: {ridx}")

    G = nx.Graph()
    # Add claim nodes (prefix 'c') and response nodes (prefix 'r') to form a bipartite graph
    claim_nodes = [f"c{cidx}" for cidx in range(len(original_claims_list))]
    response_nodes = [f"r{ridx}" for ridx in range(len(responses))]
    G.add_nodes_from(claim_nodes, bipartite=0)
    G.add_nodes_from(response_nodes, bipartite=1)
    for claim_idx, entailment_list in claim_entailment_map.items():
        for entailment_idx in entailment_list:
            G.add_edge(f"c{claim_idx}", f"r{entailment_idx}")

    claim_level_graph_metrics_dict = metrics_for_bipartite_graph(G, 0)
    return claim_level_graph_metrics_dict, claim_entailment_map, original_claims_list, G
