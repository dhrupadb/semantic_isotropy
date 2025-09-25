from typing import List, Dict, Any


SYSTEM_PROMPT = """
You are a helpful assistant that generates responses to open-ended questions.
Adhere strictly to the instructions and formatting requirements in the prompt, ensuring accuracy, consistency and correctness of formatting and content.
"""

def create_prompts(data: List[Dict[str, Any]], word_count: int = 500) -> List[str]:
    """Create prompts from the open-ended questions."""

    prompt_template = """{SYSTEM_PROMPT}

    You are given a open ended question in <question> tags.
    Write a response of approximately {word_count} words to the question. Be sure to include any topics or details you deem relevant.
    Do not include any other clarifications or context, only the response to the question.
    Start your response with the <response> tag and end with the </response> tag. No other tags should be present in the answer.
    Each tag should appear once and once only. No other tags should be present in the answer.

    <question>{question}</question>
    """

    return [prompt_template.format(SYSTEM_PROMPT=SYSTEM_PROMPT, word_count=word_count, question=question) for question in data]

