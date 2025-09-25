import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx


# Minorly modified implementation taken from: https://github.com/AlexanderVNikitin/kernel-language-entropy
# @article{nikitin2024kernel,
#   title={Kernel Language Entropy: Fine-grained Uncertainty Quantification for LLMs from Semantic Similarities},
#   author={Nikitin, Alexander and Kossen, Jannik and Gal, Yarin and Marttinen, Pekka},
#   journal={arXiv preprint arXiv:2405.20003},
#   year={2024}
# }
EPS = 1e-12

from scipy.linalg import fractional_matrix_power as fmp
def scale_entropy(entropy, n_classes):
    max_entropy = -np.log(1.0 / n_classes)  # For a discrete distribution with num_classes
    scaled_entropy = entropy / max_entropy
    return scaled_entropy

def normalize_kernel(K):
    diagonal_values = np.sqrt(np.diag(K)) + EPS
    normalized_kernel = K / np.outer(diagonal_values, diagonal_values)
    return normalized_kernel

def vn_entropy(K, normalize=True, scale=True, jitter=0):
    if normalize:
        K = normalize_kernel(K) / K.shape[0]
    result = 0
    eigvs = np.linalg.eig(K + jitter * np.eye(K.shape[0])).eigenvalues.astype(np.float64)
    for e in eigvs:
        if np.abs(e) > 1e-8:
            result -= e * np.log(e)
    if scale:
        result = scale_entropy(result, K.shape[0])
    return np.float64(result)

ALPHAS_RANGE = np.arange(0, 1.01, 0.1)
HEAT_T_RANGE = np.arange(0.1, 0.71, 0.1)
MATERN_KAPPA_RANGE = [1.0, 2.0, 3.0]
MATERN_NU_RANGE = [1.0, 2.0, 3.0]

def get_laplacian(G, norm_lapl):
    if isinstance(G, nx.DiGraph):
        L = nx.directed_laplacian_matrix(G)
    elif norm_lapl:
        L = nx.normalized_laplacian_matrix(G).toarray()
    else:
        L = nx.laplacian_matrix(G).toarray()
    return L


def heat_kernel(G: nx.Graph, t: float = 0.4, norm_lapl=False) -> torch.tensor:
    L = get_laplacian(G, norm_lapl)
    return scipy.linalg.expm(-t * L)

def matern_kernel(G: nx.Graph, kappa: float = 1, nu=1, norm_lapl=False) -> torch.tensor:
    L = get_laplacian(G, norm_lapl)
    I = np.eye(L.shape[0])
    #return fmp(nu * I + L, -alpha / 2) @ fmp(nu * I + L.T, -alpha / 2)
    return fmp((2 * nu / kappa**2) * I + L, -nu)


def get_from_sem_to_sentence_id(ordered_ids):
    from_sem_to_sentence_id = defaultdict(list)
    for i, el in enumerate(ordered_ids):
        from_sem_to_sentence_id[el].append(i)
    return from_sem_to_sentence_id


def reorder_by_semantic_ids(graph, semantic_ids, ordered_sem_ids):
    from_sem_to_sentence_id = get_from_sem_to_sentence_id(semantic_ids)
    new_graph = nx.Graph()
    for sem_id in ordered_sem_ids:
        for sent_id in from_sem_to_sentence_id[sem_id]:
            new_graph.add_node(sent_id)

    new_graph.add_edges_from(graph.edges)
    return new_graph


def get_kernels(graph):
    kernels = {}
    for t in HEAT_T_RANGE:
        kernels[f"heat_t={t:.2}"] = heat_kernel(graph, t=t)
        kernels[f"heatn_t={t:.2}"] = heat_kernel(graph, t=t, norm_lapl=True)

    for kappa in MATERN_KAPPA_RANGE:
        for nu in MATERN_NU_RANGE:
            kernels[f"matern_kappa={kappa:.2}_nu={nu:.2}"] = matern_kernel(graph, kappa=kappa, nu=nu)
            kernels[f"maternn_kappa={kappa:.2}_nu={nu:.2}"] = matern_kernel(graph, kappa=kappa, nu=nu, norm_lapl=True)

    return kernels


def all_graph_entropies(graph):
    kernels = get_kernels(graph)
    results = []
    for kernel_name, kernel in kernels.items():
        for scale in [True, False]:
            kernel_entropy = vn_entropy(kernel, scale=scale)
            postfix = "_s" if scale else ""
            results.append((f'{kernel_name}_kernel_entropy{postfix}', kernel_entropy))
    return results

def check_implication(idx1, idx2, entailment_matrix):
    activations = entailment_matrix[idx1, idx2, :]
    largest_index = np.argmax(activations)  # pylint: disable=no-member
    confidence = np.max(activations)
    prediction = largest_index.item()
    return prediction, confidence

def get_entailment_graph(strings_id_list, entailment_matrix, is_weighted=False, example=None, weight_strategy="manual"):
    """
    Get graph of entailment
    """
    def get_edge(i, j, is_weighted=False):
        implication_1, prob_impl1 = np.argmax(entailment_matrix[i, j, :]).item(), np.max(entailment_matrix[i, j, :])
        implication_2, prob_impl2 = np.argmax(entailment_matrix[j, i, :]).item(), np.max(entailment_matrix[j, i, :])
        assert (implication_1 in [0, 1, 2])
        weight = int(implication_1 == 2) + int(implication_2 == 2) + 0.5 * int(implication_1 == 1) + 0.5 * int(implication_2 == 1)
        if is_weighted:
            if weight_strategy == "manual":
                return weight
            elif weight_strategy == "deberta":
                return prob_impl1 + prob_impl2
            else:
                raise ValueError(f"Unknown weight strategy {weight_strategy}")
        return weight >= 1.5

    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_id_list)
    # Keep track of current id.
    next_id = 0
    nodes = range(len(strings_id_list))
    edges = []
    for i, string1 in enumerate(strings_id_list):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i + 1, len(strings_id_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                edge = get_edge(i, j, is_weighted=is_weighted)
                if is_weighted:
                    if edge:
                        edges.append((i, j, edge))
                else:
                    edges.append((i, j))

    G = nx.Graph()
    G.add_nodes_from(nodes)
    if is_weighted:
        G.add_weighted_edges_from(edges)
    else:
        G.add_edges_from(edges)
    return G

def get_semantic_ids(strings_id_list, entailment_matrix, strict_entailment=False, example=None):
    """Group list of predictions into semantic meaning."""

    def are_equivalent(i, j):
        implication_1, prob_impl1 = np.argmax(entailment_matrix[i, j, :]).item(), np.max(entailment_matrix[i, j, :])
        implication_2, prob_impl2 = np.argmax(entailment_matrix[j, i, :]).item(), np.max(entailment_matrix[j, i, :])
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

        if strict_entailment:
            semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)

        else:
            implications = [implication_1, implication_2]
            # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
            semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

        return semantically_equivalent

    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_id_list)
    # Keep track of current id.
    next_id = 0
    for i, string1 in enumerate(strings_id_list):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i+1, len(strings_id_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if are_equivalent(i, j):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids

    return semantic_set_ids


def get_semantic_ids_graph(string_id_list, entailment_matrix, semantic_ids, ordered_ids, strict_entailment=False, example=None):
    """Group list of predictions into semantic meaning."""
    def are_similar(i, j, is_weighted=False):
        implication_1, prob_impl1 = np.argmax(entailment_matrix[i, j, :]).item(), np.max(entailment_matrix[i, j, :])
        implication_2, prob_impl2 = np.argmax(entailment_matrix[j, i, :]).item(), np.max(entailment_matrix[j, i, :])
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

        return (implication_1 == 2) + (implication_1 == 1) * 0.5 +\
               (implication_2 == 2) + (implication_2 == 1) * 0.5

    # Initialise all ids with -1.
    nodes = ordered_ids
    weights = defaultdict(list) # (i, j) -> weight
    for i, string1 in enumerate(strings_id_list):
        node_i = semantic_ids[i]
        for j in range(i + 1, len(strings_id_list)):
            node_j = semantic_ids[j]
            edge_weight = are_similar(i, j)
            if edge_weight > 0:
                weights[(node_i, node_j)].append(edge_weight)
    for k, v in weights.items():
        weights[k] = np.sum(v)
    assert -1 not in semantic_ids
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from([(i, j, w) for (i, j), w in weights.items()])
    return G

def calc_kle(mat):
    id_list = list(range(mat.shape[0]))
    entropies = defaultdict(list)

    semantic_ids = get_semantic_ids(id_list, mat)

    graph = get_entailment_graph(id_list, mat)
    for k, value in all_graph_entropies(graph):
        entropies[k].append(value)

    weighted_graph = get_entailment_graph(id_list, mat)
    for k, value in all_graph_entropies(weighted_graph):
        entropies[f"weighted_{k}"].append(value)
    return entropies, semantic_ids

def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from how often different clusters get assigned.

    We estimate the categorical distribution over cluster assignments from the
    semantic ids. The uncertainty is then given by the entropy of that
    distribution. This estimate does not use token likelihoods, it relies soley
    on the cluster assignments. If probability mass is spread of between many
    clusters, entropy is larger. If probability mass is concentrated on a few
    clusters, entropy is small.

    Input:
        semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
    Output:
        cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
    """

    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts/n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy

# Taken from https://github.com/jlko/semantic_uncertainty (with minor modifications)

def are_equivalent(i, j, entailment_matrix, strict_entailment=False, example=None):
    implication_1 = np.argmax(entailment_matrix[i, j, :])
    implication_2 = np.argmax(entailment_matrix[j, i, :])
    assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

    if strict_entailment:
        semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)

    else:
        implications = [implication_1, implication_2]
        # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
        semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

    return semantically_equivalent


def get_semantic_ids(entailment_matrix, strict_entailment=False, example=None):
    """Group list of predictions into semantic meaning."""
    # Initialise all ids with -1.
    semantic_set_ids = [-1] * entailment_matrix.shape[0]
    # Keep track of current id.
    next_id = 0
    for i in range(entailment_matrix.shape[0]):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i+1, entailment_matrix.shape[0]):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if are_equivalent(i, j, entailment_matrix, strict_entailment, example):
                    semantic_set_ids[j] = next_id
            next_id += 1
    assert -1 not in semantic_set_ids
    return semantic_set_ids

def logsumexp_by_id(semantic_ids, log_likelihoods, agg='sum'):
    """Sum probabilities with the same semantic id.

    Log-Sum-Exp because input and output probabilities in log space.
    """
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
    log_likelihood_per_semantic_id = []

    for uid in unique_ids:
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        if agg == 'sum':
            logsumexp_value = np.log(np.sum(np.exp(id_log_likelihoods))) - 5.0
        elif agg == 'sum_normalized':
            log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        elif agg == 'mean':
            logsumexp_value = np.log(np.mean(np.exp(id_log_likelihoods)))
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)

    return log_likelihood_per_semantic_id


def predictive_entropy(log_probs):
    """Compute MC estimate of entropy.

    `E[-log p(x)] ~= -1/N sum_i log p(x_i)` where i are the is the sequence
    likelihood, i.e. the average token likelihood.
    """

    entropy = -np.sum(log_probs) / len(log_probs)
    return entropy


def predictive_entropy_rao(log_probs):
    entropy = -np.sum(np.exp(log_probs) * log_probs)
    return entropy


def get_entailment_matrix(responses, model, tokenizer, device):
    entailment_matrix = torch.zeros((len(responses), len(responses), 3))
    for i, response1 in enumerate(responses):
        for j, response2 in enumerate(responses):
            if i == j:
                entailment_matrix[i, j, :] = torch.tensor([0, 0, 1.0])
            else:
                inputs = tokenizer(response1['response'], response2['response'], return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    predictions = F.softmax(logits, dim=1)
                    if predictions.shape[1] == 3:
                        entailment_matrix[i, j, :] = predictions
                    elif predictions.shape[1] == 2: # potsawee/deberta-v3-large-mnli
                        entailment_matrix[i, j, 2] = predictions[0][0]
                        entailment_matrix[i, j, 0] = predictions[0][1]
                        entailment_matrix[i, j, 1] = 1.0 - (torch.sum(predictions[0]))
                    else:
                        raise RuntimeError(f"Unexpected number of predictions: {predictions.shape}")

    return entailment_matrix.cpu().numpy()
#########################

def graphLaplacian(entailment_matrix, pct_k = 0.5):
    m = entailment_matrix.shape[0]
    E = entailment_matrix[:, :, 2]
    W = (E + E.T)/2
    D = np.eye(W.shape[0]) * W.sum(axis=1)
    D_inv = np.linalg.inv(D)
    D_inv_sqrt = np.linalg.cholesky(D_inv)
    L = np.eye(D.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    eigvals = np.linalg.eigvals(L)
    U_eigv = np.sum(np.maximum(0, 1 - eigvals))
    U_deg = np.trace(m*np.eye(m) - D)/(m**2)
    k = int(L.shape[0]*pct_k)
    L_eigvals, L_eigvecs = np.linalg.eig(L)
    smallest_k_eigvec = L_eigvecs[:k]
    U_ecc = np.linalg.norm((smallest_k_eigvec - smallest_k_eigvec.mean(axis=1).reshape(-1, 1)).T)
    return U_eigv, U_deg, U_ecc, L

def SelfCheckGPT_NLI(entailment_matrix):
    expe = np.exp(entailment_matrix[:, :, 2]) # entailment
    expc = np.exp(entailment_matrix[:, :, 0]) # contradiction
    p_contra = expc / (expc + expe)
    S_NLI = p_contra.mean(axis=1)
    U_NLI = np.mean(1 - S_NLI)
    return U_NLI, S_NLI

def entailment_metrics(responses, model, tokenizer, device):
    entailment_matrix = get_entailment_matrix(responses, model, tokenizer, device)
    semantic_ids = get_semantic_ids(entailment_matrix)
    semantic_entropy = cluster_assignment_entropy(semantic_ids)
    U_eigv, U_deg, U_ecc, L = graphLaplacian(entailment_matrix)
    U_NLI, S_NLI = SelfCheckGPT_NLI(entailment_matrix)
    entailment_metrics = {'semantic_entropy': {'semantic_entropy': semantic_entropy, 'semantic_ids': semantic_ids},
                          'graph_metrics': {'U_eigv': U_eigv, 'U_deg': U_deg, 'U_ecc': U_ecc, 'Laplacian': L},
                          'SelfCheckGPT_NLI': {'U_NLI': U_NLI, 'S_NLI': S_NLI}
                          }
    return entailment_matrix, entailment_metrics
