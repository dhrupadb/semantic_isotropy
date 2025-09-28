import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def get_entity_page_idx(entity: str, trivia_entry: Dict[str, Any]) -> int:
    """
    Get the reference doc page index of an entity in a trivia entry
    """
    valid_pages = [(i, title) for i, title in enumerate(trivia_entry['entity_pages']['title']) if title.lower().strip() == entity.lower().strip()]
    if len(valid_pages) == 0:
        logger.warning(f"No valid pages found for entity \"{entity}\" in trivia entry {trivia_entry['question_id']}. Using the first page.")
        return -1
    if len(valid_pages) > 1:
        logger.warning(f"Multiple valid pages found for entity \"{entity}\" in trivia entry {trivia_entry['question_id']}. Using the first page.")
        return valid_pages[0][0]
    return valid_pages[0][0]

def strip_and_return(string):
    left_stripped = string.lstrip()
    right_stripped = left_stripped.rstrip()

    left_trim = string[:len(string) - len(left_stripped)]
    right_trim = left_stripped[len(right_stripped):]

    return right_stripped, left_trim, right_trim
