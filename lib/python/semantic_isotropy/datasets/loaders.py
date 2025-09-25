import logging
import json
import pickle
import pandas as pd
from typing import List, Dict, Any
import os
import sqlite3

from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_triviaqa():
    """Load TriviaQA Wikipedia dataset"""
    logger.info("Loading TriviaQA dataset")
    return load_dataset("trivia_qa", "rc.wikipedia")

def load_factscore(factscore_db_path=None):
    """Load FactScore-Bio Wikipedia dataset"""
    logger.info("Loading FactScore-Bio dataset")

    # Get the current directory properly
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bio_entries_path = os.path.join(\
        os.path.dirname(\
        os.path.dirname(\
        os.path.dirname(os.path.dirname(current_dir)))),
        "data", "bio_entities.txt")

    with open(bio_entries_path, "r") as f:
        entities = [line.strip() for line in f.readlines()]

    # 4. Open a sqlite connection
    factscore_db_path = f"/Users/{os.environ.get('USER')}/datasets/experiments/semantic_isotropy/factscore/enwiki-20230401.db" if not factscore_db_path else factscore_db_path
    conn = sqlite3.connect(factscore_db_path)
    cursor = conn.cursor()

    # Fix the SQL query to properly quote the entity names
    placeholders = ', '.join(['?' for _ in entities])
    cursor.execute(f"SELECT title, text FROM documents WHERE title IN ({placeholders})", entities)

    results = cursor.fetchall()
    cursor.close()
    conn.close()
    results = {title: text for title, text in results}
    return results

def load_data(input_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSON or pickle file"""
    logger.info(f"Loading data from {input_path}")
    if input_path.endswith('.json'):
        ftype = 'json'
    elif input_path.endswith('.jsonl'):
        ftype = 'jsonl'
    elif input_path.endswith('.pkl'):
        ftype = 'pickle'
    else:
        raise ValueError(f"Unsupported file type: {input_path}")

    if ftype == 'json':
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif ftype == 'jsonl':
        with open(input_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    elif ftype == 'pickle':
        with open(input_path, 'rb') as f:
            data = pickle.load(f)

    return data

def load_csv(input_path: str) -> pd.DataFrame:
    """
    Load and optionally sample from the input dataset.

    Args:
        input_path: Path to the input CSV file

    Returns:
        List of dictionary examples
    """
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)

    return df
