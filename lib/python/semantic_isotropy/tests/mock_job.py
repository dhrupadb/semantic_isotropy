"""
Mock job script for testing the metric runner.
Simulates a realistic ML job with stochastic runtime, device usage, and various exit conditions.
"""

import argparse
import time
import random
import sys
from pathlib import Path
import json
import datetime

def simulate_device_load(device: str, duration: float):
    """Simulate device-specific workload"""
    if device == 'auto' or 'cuda' in device:
        # Simulate GPU workload - more variable timing
        base_time = duration
        variation = random.uniform(0.8, 1.3)  # ±30% variation for GPU work
        actual_time = base_time * variation

        print(f"[GPU WORK] Initializing CUDA on {device}")
        time.sleep(0.5)  # Simulate GPU setup
        print(f"[GPU WORK] Loading model to GPU memory...")
        time.sleep(1.0)  # Simulate model loading
        print(f"[GPU WORK] Processing batch data...")

        # Simulate chunked processing
        chunks = max(1, int(actual_time / 2))
        chunk_time = actual_time / chunks

        for i in range(chunks):
            time.sleep(chunk_time)
            progress = ((i + 1) / chunks) * 100
            print(f"[GPU WORK] Progress: {progress:.1f}% complete")

            # Simulate occasional GPU memory warnings
            if random.random() < 0.1:
                print(f"[GPU WORK] Warning: High GPU memory usage detected")

        print(f"[GPU WORK] GPU processing complete")

    elif device == 'mps':
        # Simulate MPS workload - Apple Silicon specific
        actual_time = duration * random.uniform(0.9, 1.2)  # ±20% variation

        print(f"[MPS WORK] Initializing Metal Performance Shaders")
        time.sleep(0.3)
        print(f"[MPS WORK] Optimizing for Apple Silicon...")

        # Simulate MPS processing
        steps = max(1, int(actual_time / 1.5))
        step_time = actual_time / steps

        for i in range(steps):
            time.sleep(step_time)
            progress = ((i + 1) / steps) * 100
            print(f"[MPS WORK] Neural engine progress: {progress:.1f}%")

        print(f"[MPS WORK] MPS processing complete")

    elif device == 'cpu':
        # Simulate CPU workload - more predictable timing
        actual_time = duration * random.uniform(0.95, 1.1)  # ±10% variation

        print(f"[CPU WORK] Starting CPU-based processing...")
        print(f"[CPU WORK] Using multi-threading optimization")

        # Simulate CPU processing
        iterations = max(1, int(actual_time / 1.0))
        iter_time = actual_time / iterations

        for i in range(iterations):
            time.sleep(iter_time)
            progress = ((i + 1) / iterations) * 100
            print(f"[CPU WORK] CPU threads progress: {progress:.1f}%")

            # Simulate CPU resource warnings
            if random.random() < 0.05:
                print(f"[CPU WORK] Note: High CPU utilization")

        print(f"[CPU WORK] CPU processing complete")

    else:
        # Fallback processing
        actual_time = duration * random.uniform(0.9, 1.1)
        print(f"[GENERIC WORK] Processing on device: {device}")
        time.sleep(actual_time)
        print(f"[GENERIC WORK] Processing complete")

def simulate_failure_scenarios(failure_rate: float = 0.05):
    """Randomly simulate realistic failure scenarios"""
    if random.random() < failure_rate:
        failure_types = [
            ("CUDA_OUT_OF_MEMORY", "CUDA out of memory error", 2),
            ("MODEL_LOAD_ERROR", "Failed to load embedding model", 1),
            ("DATA_CORRUPTION", "Input data appears corrupted", 1),
            ("TIMEOUT", "Operation timed out", 3),
        ]

        failure_type, error_msg, exit_code = random.choice(failure_types)

        print(f"[ERROR] {error_msg}", file=sys.stderr)
        print(f"[ERROR] Failure type: {failure_type}", file=sys.stderr)
        sys.exit(exit_code)

def main():
    parser = argparse.ArgumentParser(description="Mock job for testing metric runner")

    # Core parameters that real gen_metric.py would have
    parser.add_argument('--input-path', required=True, help='Input data path')
    parser.add_argument('--output-path', required=True, help='Output file path')
    parser.add_argument('--metric', required=True, help='Metric type')
    parser.add_argument('--embedding-model', required=True, help='Embedding model name')
    parser.add_argument('--response-count', type=int, required=True, help='Number of responses')
    parser.add_argument('--response-max', type=int, help='Maximum number of responses (ignored in mock)')
    parser.add_argument('--device', default='auto', help='Device to use')
    parser.add_argument('--pooling', help='Pooling method')

    # Test-specific parameters
    parser.add_argument('--min-runtime', type=float, default=5.0, help='Minimum runtime in seconds')
    parser.add_argument('--max-runtime', type=float, default=20.0, help='Maximum runtime in seconds')
    parser.add_argument('--failure-rate', type=float, default=0.05, help='Probability of simulated failure')

    args = parser.parse_args()

    # Print job startup info
    start_time = datetime.datetime.now()
    print(f"[MOCK JOB] Started at {start_time}")
    print(f"[MOCK JOB] Metric: {args.metric}")
    print(f"[MOCK JOB] Model: {args.embedding_model}")
    print(f"[MOCK JOB] Response count: {args.response_count}")
    print(f"[MOCK JOB] Device: {args.device}")
    print(f"[MOCK JOB] Pooling: {args.pooling}")
    print(f"[MOCK JOB] Input: {args.input_path}")
    print(f"[MOCK JOB] Output: {args.output_path}")

    # Check for failure scenarios early
    simulate_failure_scenarios(args.failure_rate)

    # Simulate loading phase
    print(f"[MOCK JOB] Loading input data...")
    time.sleep(random.uniform(0.5, 2.0))

    # Simulate input validation
    print(f"[MOCK JOB] Validating input parameters...")
    time.sleep(random.uniform(0.2, 1.0))

    # Check for failure during validation
    simulate_failure_scenarios(args.failure_rate * 0.5)

    # Determine runtime (stochastic)
    runtime = random.uniform(args.min_runtime, args.max_runtime)
    print(f"[MOCK JOB] Estimated processing time: {runtime:.1f} seconds")

    # Simulate the main workload based on device
    simulate_device_load(args.device, runtime)

    # Check for failure during processing
    simulate_failure_scenarios(args.failure_rate * 0.3)

    # Simulate output generation
    print(f"[MOCK JOB] Generating output...")

    # Create output directory if it doesn't exist
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create realistic output file with job metadata
    result = {
        "metric": args.metric,
        "embedding_model": args.embedding_model,
        "response_count": args.response_count,
        "device": args.device,
        "pooling": args.pooling,
        "runtime_seconds": runtime,
        "start_time": start_time.isoformat(),
        "end_time": datetime.datetime.now().isoformat(),
        "status": "completed",
        "results": {
            "score": random.uniform(0.1, 0.9),  # Mock metric score
            "variance": random.uniform(0.01, 0.1),
            "samples_processed": args.response_count
        }
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    time.sleep(random.uniform(0.2, 0.8))  # Simulate file I/O

    end_time = datetime.datetime.now()
    total_runtime = (end_time - start_time).total_seconds()

    print(f"[MOCK JOB] Job completed successfully!")
    print(f"[MOCK JOB] Total runtime: {total_runtime:.2f} seconds")
    print(f"[MOCK JOB] Output saved to: {args.output_path}")
    print(f"[MOCK JOB] Finished at {end_time}")

if __name__ == "__main__":
    main()
