import inspect
import os
import json
import pickle
import click
import datetime
import logging
from pathlib import Path
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)


def save_config(output_dir: str, ctx: click.Context, dryrun: bool):
    """Save configuration parameters to a timestamped file in the output directory"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Get the calling script's filename
    frame = inspect.stack()[1]
    calling_script = os.path.splitext(os.path.basename(frame.filename))[0]
    config_path = os.path.join(output_dir, f'{calling_script}_params_{timestamp}{".cfg" if not dryrun else ".dryrun.cfg"}')

    # Automatically get all parameters from Click's context
    config = {
        k: v for k, v in ctx.params.items()
    }
    config['timestamp'] = datetime.datetime.now().isoformat()

    try:
        if not dryrun:
            os.makedirs(output_dir, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved configuration to {config_path}")
        else:
            logger.info(f"DRY RUN: Would save configuration to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")


def write_results(output_path: str, result_map: Dict[str, Any], coalesce_func: Callable = lambda x, y: y): # Just return the result map if no coalesce function is provided
    """Write results to output JSON file with atomic writes"""
    temp_path = output_path + '.tmp'
    if output_path.endswith('.json'):
        ftype = 'json'
    elif output_path.endswith('.jsonl'):
        ftype = 'jsonl'
    elif output_path.endswith('.pkl'):
        ftype = 'pickle'
    else:
        raise ValueError(f"Unsupported file type: {output_path}")

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load existing results
        existing_results = []
        if Path(output_path).exists():
            if ftype == 'json':
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
            elif ftype == 'jsonl':
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_results = [json.loads(line) for line in f]
            elif ftype == 'pickle':
                with open(output_path, 'rb') as f:
                    existing_results = pickle.load(f)
        
        coalesced_results = coalesce_func(existing_results, result_map)

        # Write atomically using temporary file
        if ftype == 'json':
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(coalesced_results, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
        elif ftype == 'jsonl':
            with open(temp_path, 'w', encoding='utf-8') as f:
                for item in coalesced_results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
        elif ftype == 'pickle':
            with open(temp_path, 'wb') as f:
                pickle.dump(coalesced_results, f)
                f.flush()
                os.fsync(f.fileno())

        os.replace(temp_path, output_path)  # atomic operation
        logger.info(f"Wrote results to {output_path}")

    except Exception as e:
        logger.error(f"Error writing results to {output_path}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

def init_logger(name: str, level: str = 'INFO'):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO if level == 'INFO' else logging.DEBUG if level == 'DEBUG' else logging.WARNING if level == 'WARNING' else logging.ERROR if level == 'ERROR' else logging.CRITICAL if level == 'CRITICAL' else logging.FATAL if level == 'FATAL' else logging.NOTSET,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(name)