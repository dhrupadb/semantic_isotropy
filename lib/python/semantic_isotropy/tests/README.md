# Metric Runner Test Suite

This directory contains a comprehensive test suite for the metric runner system, designed to validate device allocation, job scheduling, logging, and error handling functionality.

## Test Components

### 1. Mock Job Script (`mock_job.py`)
- Simulates realistic ML jobs with stochastic runtime (5-20 seconds by default)
- Supports all device types: CUDA, MPS, CPU
- Includes realistic failure scenarios and device-specific behavior
- Generates proper output files with job metadata

### 2. Test Configurations

#### `test_runs_comprehensive.cfg`
- **~51 test jobs** across 5 different scenarios
- Tests mixed GPU requirements (1-GPU, 2-GPU, all-GPU jobs)
- CPU concurrency testing with configurable limits
- MPS sequential execution testing
- Complex priority scenarios and stress testing

#### `test_device_scenarios.cfg`
- **Device-specific edge cases** and contention scenarios
- GPU resource contention with limited hardware
- Priority queue testing (all-GPU jobs run last)
- CPU concurrency limits
- MPS sequential behavior validation
- Failure handling and resilience testing

### 3. Test Runner (`run_comprehensive_tests.py`)
- Automated test execution and monitoring
- Dry run validation of device allocation logic
- Full execution testing with progress monitoring
- Results validation and performance benchmarking
- Comprehensive test reporting

### 4. Results Validator (`validate_test_results.py`)
- Post-execution validation of job outputs
- Device allocation pattern analysis
- Logging functionality verification
- Performance metrics calculation
- Recommendation generation

## Quick Start

### Run Basic Tests (Dry Run Only)
```bash
cd /path/to/semantic_isotropy/lib/python/longform_uq/tests/
./run_comprehensive_tests.py --dry-run-only
```

### Run Full Test Suite
```bash
./run_comprehensive_tests.py --max-runtime 15
```

### Run Device-Specific Tests
```bash
# Test with device scenarios config
python ../../../../scripts/runner.py --config test_device_scenarios.cfg --dryrun
```

### Validate Results
```bash
./validate_test_results.py /tmp/metric_runner_tests
```

## Test Scenarios Explained

### 1. **Device Allocation Testing**
- **Single GPU Jobs**: Multiple jobs competing for individual GPUs
- **Multi-GPU Jobs**: Jobs requiring 2+ GPUs waiting for resources
- **All-GPU Jobs**: Jobs requiring all GPUs (lowest priority)
- **GPU Reuse**: Ensuring GPUs are properly allocated and released

### 2. **Priority System Testing**  
- **Priority 1**: 1-2 GPU jobs (run first)
- **Priority 2**: 3+ GPU jobs (run second)
- **Priority 3**: All-GPU jobs (run last)

### 3. **Device Type Testing**
- **CUDA**: Parallel execution with GPU allocation
- **MPS**: Sequential execution (Apple Silicon)
- **CPU**: Parallel execution with concurrency limits

### 4. **Error Handling Testing**
- Configurable failure rates (0-30%)
- Realistic error scenarios (CUDA OOM, model loading failures, etc.)
- Graceful degradation and job isolation

### 5. **Performance Testing**
- Job throughput measurement
- Resource utilization tracking
- Runtime distribution analysis

## Expected Behavior

### ✅ Successful Test Run Should Show:

1. **Device Allocation**:
   ```
   🟢 job_01: [Device: cuda:0] STARTED
   🟢 job_02: [Device: cuda:1] STARTED  
   🟢 job_03: [Device: cuda:2] STARTED
   🔴 job_04: WAITING for resources
   ```

2. **Priority Ordering**:
   - Single-GPU jobs start immediately
   - All-GPU jobs wait for all resources
   - Proper queueing when resources unavailable

3. **Output Files**:
   - One `.json` file per successful job
   - Contains job metadata, runtime, and mock results

4. **Log Files**:
   - Separate stdout/stderr logs per job
   - Organized by job ID and timestamp

### 🔍 Test Validation Checks:

- ✅ Job success rate ≥ 90%
- ✅ Proper device allocation patterns
- ✅ Log file completeness
- ✅ Output file validity
- ✅ Performance within expected ranges

## Configuration Parameters

### Job Runtime Control:
```yaml
min_runtime: 3.0    # Minimum job duration (seconds)
max_runtime: 12.0   # Maximum job duration (seconds)  
failure_rate: 0.02  # Probability of simulated failure (0-1)
```

### Device Configuration:
```yaml
available_gpus: 4   # Number of GPUs to simulate
devices:
  default:
    type: "cuda"    # Device type: cuda/mps/cpu
    count: 1        # GPUs per job (for CUDA)
    max_concurrency: 4  # Concurrent jobs (for CPU)
```

## Troubleshooting

### Common Issues:

1. **All jobs get cuda:0**: 
   - Check `available_gpus` setting in config
   - Verify device allocation simulation logic

2. **Tests timeout**:
   - Increase `--max-runtime` parameter
   - Check for hanging jobs in logs

3. **Low success rate**:
   - Reduce `failure_rate` in test config
   - Check stderr logs for error details

4. **Missing output files**:
   - Verify test root directory permissions
   - Check script_path in configuration

### Debug Mode:
```bash
# Run with verbose output
python ../../../../scripts/runner.py --config test_runs_comprehensive.cfg --dryrun | tee debug.log

# Check specific job logs
ls /tmp/metric_runner_tests/logs/
cat /tmp/metric_runner_tests/logs/job_*.stderr.log
```

## Test Duration Estimates

- **Dry Run**: < 30 seconds
- **Basic Test Suite**: 5-15 minutes  
- **Comprehensive Tests**: 10-25 minutes
- **Device Scenarios**: 8-20 minutes

Times depend on configured job runtimes and number of jobs.

## Custom Test Development

To create new test scenarios:

1. Copy an existing `.cfg` file
2. Modify the `runs` section with your test cases
3. Adjust `min_runtime`, `max_runtime`, and `failure_rate`
4. Set appropriate device configurations
5. Run with the metric runner

Example minimal test:
```yaml
runs:
  - input_path: "{{root_dir}}/data/my_test.json"
    metric: "my_metric"
    embedding_model:
      - ["test/model", "mean", "test_model"]
    response_count: [100]
    min_runtime: 2.0
    max_runtime: 5.0
    devices:
      default:
        type: "cuda"
        count: 1
```

## Integration with CI/CD

The test suite returns appropriate exit codes:
- `0`: All tests passed
- `1`: Some tests failed  
- `2`: Critical failures

Example GitHub Actions integration:
```yaml
- name: Run Metric Runner Tests
  run: |
    cd scripts/segscore/tests/
    ./run_comprehensive_tests.py --max-runtime 10
```