import click
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import csv
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

from longform_uq.llm.api import chat_api as query_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(input_path: str) -> pd.DataFrame:
    """Load the output data from oeq_sample.py"""
    logger.info(f"Loading data from {input_path}")
    return pd.read_csv(input_path)

def load_triviaqa():
    """Load TriviaQA Wikipedia dataset"""
    logger.info("Loading TriviaQA dataset")
    return load_dataset("trivia_qa", "rc.wikipedia")

def create_prompt(reference_doc: str, response: str) -> str:
    """Create evaluation prompt from reference document and response"""
    return f"""
You are an NLP segmentation and evaluation engine. Examine the scenario below. You are given:
1. The name of an entity / person / place / thing etc. in <entity> tags.
1. A reference document regarding the entity in <reference_doc> tags.
3. A response about the entity to evaluate in <response> tags.

Your Tasks:
1. Segment <response> into individual statements. Do not modify or paraphrase the original text. Each statement may be a phrase or sentence that conveys a single piece of information about the <entity>. Give each statement in <statement> tags and enclose the statements in <statements> tags . 
2. For each statement, classify it based on its factual accuracy given the context of the `reference_doc`. Your judgement should be based on the reference document only and should not be based on any other information. The classes are described as follows:
    - `Likely False`: It is highly likely / certain that the fact is false or incorrect.
    - `Probably False`: There is some non-trivial chance that the fact is false or incorrect.
    - `Uncertain`: There is no clear evidence for the fact being true or false or it is not possible to determine the factual accuracy of the fact based on the reference document.
    - `Probably True`: The fact is probably true given the reference document but not certain.
    - `Likely True`: The fact is highly likely / definitelytrue given the reference document.

Examples:
##### EXAMPLE 1 ######
<entity>
London, UK
</entity>

Reference Document:
<reference_doc>
London, England's capital, boasts a rich history spanning millennia. Founded by the Romans as Londinium around 47 AD, it became a major port and trading center. After the Roman withdrawal, Anglo-Saxons established Lundenwic, which later fell to Viking raids.  The Norman Conquest in 1066 led to the construction of the Tower of London, a symbol of royal power. London thrived during the medieval period, becoming a major center for trade, finance, and culture. It weathered plagues, fires, and civil wars, emerging as a global metropolis and the heart of the British Empire. Today, London remains a vibrant hub, blending its historical legacy with modern dynamism.
</reference_doc>

Response to evaluate:
<response>
London, the capital city of England and the United Kingdom, is a vibrant metropolis steeped in history and brimming with modern energy. With a population of over 9 million people, it stands as one of the world's most influential global cities, known for its diverse culture, iconic landmarks, and rich heritage.

The city's history stretches back over three millennia, founded by the Romans as Londinium in 43 AD. Throughout the centuries, London has played a pivotal role in world affairs, serving as the heart of the British Empire and surviving tumultuous events such as the Great Fire of 1666 and the Blitz during World War I.

Today, London is a melting pot of cultures, with over 300 languages spoken within its boundaries. This diversity is reflected in its neighborhoods, each with its own unique character and charm. From the trendy streets of Shoreditch to the upscale boutiques of Mayfair, there's something for everyone in this cosmopolitan city.
</response>

Segmented and classified response:
<statements>
<statement>London, the capital city of England and the United Kingdom</statement> `Likely True`
<statement>is a vibrant metropolis steeped in history and brimming with modern energy</statement> `Likely True`
<statement>With a population of over 9 million people</statement> `Uncertain`
<statement>it stands as one of the world's most influential global cities, known for its diverse culture, iconic landmarks, and rich heritage.</statement> `Probably True`
<statement>The city's history stretches back over three millennia, founded by the Romans as Londinium in 43 BC</statement> `Likely False`
<statement>Throughout the centuries, London has played a pivotal role in world affairs, serving as the heart of the British Empire</statement> `ProbablyTrue`
<statement>and surviving tumultuous events such as the Great Fire of 1666</statement> `Probably True`
<statement>and the Blitz during World War I</statement `Likely False`
<statement>Today, London is a melting pot of cultures, with over 300 languages spoken within its boundaries</statement> `Uncertain`
<statement>This diversity is reflected in its neighborhoods, each with its own unique character and charm.</statement> `Uncertain`
<statement>From the trendy streets of Shoreditch to the upscale boutiques of Mayfair, there's something for everyone in this cosmopolitan city</statement> `Probably True`
</statements>

########################
##### EXAMPLE 2 ######
Reference Document:
<reference_doc>
Obsidian is a naturally occurring volcanic glass formed from rapidly cooling lava. Its glassy texture and conchoidal fracture result from minimal crystal growth during the cooling process. Typically jet-black, obsidian can also appear red, brown, or even iridescent due to the presence of mineral inclusions.

Prized for its sharpness and beauty since ancient times, obsidian was used for tools, weapons, and ornaments.  Its glassy nature made it ideal for crafting arrowheads, knives, and mirrors. Today, obsidian remains popular in jewelry and decorative objects.

Found in volcanic regions worldwide, obsidian provides valuable insights into volcanic activity and Earth's geological processes.
</reference_doc>

Response to evaluate:
<response>
Obsidian is a naturally occurring volcanic glass formed when lava cools rapidly, preventing the formation of crystalline structures. Its amorphous, non-crystalline structure gives it a smooth, homogeneous texture, making it distinct from most igneous rocks. Obsidian is remarkably brittle yet strong, with a Mohs hardness of about 1-2. Its unique fracture pattern, known as conchoidal fracturing, allows it to be shaped into extremely sharp edges, sharper than even modern steel surgical scalpels. This quality made obsidian a vital material for crafting tools and weapons in ancient cultures and continues to find use in precision cutting applications in modern surgery.
</response>

Segmented and classified response:
<statements>
<statement>Obsidian is a naturally occurring volcanic glass formed when lava cools rapidly, preventing the formation of crystalline structures</statement> `Likely True`
<statement>Its amorphous, non-crystalline structure gives it a smooth, homogeneous texture, making it distinct from most igneous rocks</statement> `Probably True`
<statement>Obsidian is remarkably brittle yet strong,<statement> `Probably True`
<statement>with a Mohs hardness of about 1-2</statement> `Uncertain`
<statement>Its unique fracture pattern, known as conchoidal fracturing, allows it to be shaped into extremely sharp edges, sharper than even modern steel surgical scalpels</statement> `Likely True`
<statement>This quality made obsidian a vital material for crafting tools and weapons in ancient cultures</statement> `Likely True`
<statement>and continues to find use in precision cutting applications in modern surgery</statement> `Uncertain`
</statements>

########################

Reference Document:
<reference_doc>
{reference_doc}
</reference_doc>

Response to Evaluate:
<response>
{response}
</response>
"""

def write_results(output_path: str, results: List[Dict[str, Any]]):
    """Write results to output CSV file, creating if doesn't exist"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mode = 'a' if Path(output_path).exists() else 'w'
    
    with open(output_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f,
                              fieldnames=results[0].keys(),
                              quoting=csv.QUOTE_ALL,
                              escapechar='\\',
                              doublequote=True)
        
        if mode == 'w':
            writer.writeheader()
        
        writer.writerows(results)
        f.flush()  # Flush after each write

@click.command()
@click.option('--input-path', required=True, help='Path to the input CSV file of generated open ended responses')
@click.option('--output-path', required=True, help='Output path for the evaluation results')
@click.option('--batch-size', default=20, help='Number of entries to process before writing')
def main(input_path: str, output_path: str, batch_size: int):
    """Evaluate responses against TriviaQA reference documents"""
    
    # Load data
    df = load_data(input_path)
    triviaqa = load_triviaqa()
    
    # Create unique key and group data
    df['unique_key'] = df['index'].astype(str) + '_' + df['idx_cat']
    grouped = df.groupby('unique_key')
    
    results_buffer = []
    
    for key, group in tqdm(grouped):
        index = group['index'].iloc[0]
        idx_cat = group['idx_cat'].iloc[0]
        
        # Select appropriate dataset split
        dataset_split = triviaqa['train'] if idx_cat == 'train' else triviaqa['validation']
        
        try:
            # Get reference document from TriviaQA
            trivia_entry = dataset_split[index]
            reference_doc = trivia_entry['entity_pages']['wiki_context'][0]  # Get first wiki context
            
            # Process each response in the group
            for _, row in group.iterrows():
                prompt = create_prompt(reference_doc, row['response'])
                api_result = query_api(prompt)
                
                result = {
                    'index': row['index'],
                    'idx_cat': row['idx_cat'],
                    'open_ended_question': row['open_ended_question'],
                    'response': row['response'],
                    'evaluation': api_result
                }
                
                results_buffer.append(result)
                
                # Write results when buffer reaches batch size
                if len(results_buffer) >= batch_size:
                    write_results(output_path, results_buffer)
                    results_buffer = []
                    
        except Exception as e:
            logger.error(f"Error processing key {key}: {e}")
            continue
    
    # Write any remaining results
    if results_buffer:
        write_results(output_path, results_buffer)
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()
