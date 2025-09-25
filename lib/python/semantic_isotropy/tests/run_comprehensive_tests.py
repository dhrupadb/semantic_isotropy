"""
Comprehensive Test Suite for Metric Runner

This script sets up and runs comprehensive tests for the metric runner system,
testing device allocation, job scheduling, logging, and error handling.
"""

import subprocess
import sys
import shutil
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import argparse


class MetricRunnerTestSuite:
    """Comprehensive test suite for metric runner"""

    def __init__(self, test_root: str = "/tmp/metric_runner_tests"):
        self.test_root = Path(test_root)
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent.parent.parent.parent
        self.results = {
            "tests_run": [],
            "total_jobs": 0,
            "successful_jobs": 0,
            "failed_jobs": 0,
            "test_results": {},
            "device_allocation_tests": {},
            "performance_metrics": {}
        }

    def setup_test_environment(self):
        """Set up test directories and mock data files"""
        print("🔧 Setting up test environment...")

        # Clean and create test directories
        if self.test_root.exists():
            shutil.rmtree(self.test_root)

        self.test_root.mkdir(parents=True, exist_ok=True)
        (self.test_root / "data").mkdir(exist_ok=True)
        (self.test_root / "results").mkdir(exist_ok=True)
        (self.test_root / "logs").mkdir(exist_ok=True)

        # Create mock input data files
        mock_data_files = [
            "sample_input_1.json",
            "sample_input_2.json",
            "sample_input_3.json",
            "sample_input_4.json",
            "sample_input_5.json"
        ]

        for data_file in mock_data_files:
            mock_data = {
                "responses": [
                    {"id": i, "text": f"Mock response {i} for testing", "score": 0.5}
                    for i in range(1000)  # Plenty of mock responses
                ],
                "metadata": {
                    "source": "test_suite",
                    "created": time.time(),
                    "file": data_file
                }
            }

            with open(self.test_root / "data" / data_file, 'w') as f:
                json.dump(mock_data, f, indent=2)

        print(f"✅ Test environment set up in {self.test_root}")

    def run_dry_run_test(self) -> Dict[str, Any]:
        """Test dry run mode to validate device allocation logic"""
        print("\n🧪 Running dry run test...")

        cmd = [
            sys.executable,
            str(self.project_root / "scripts" / "runner.py"),
            "--config", str(self.script_dir / "test_runs_comprehensive.cfg"),
            "--dryrun"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # Dry run should be fast
            )

            dry_run_result = {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "device_allocations": self._parse_device_allocations(result.stdout)
            }

            if dry_run_result["success"]:
                print("✅ Dry run test passed")
                self._validate_device_allocation_logic(dry_run_result["device_allocations"])
            else:
                print("❌ Dry run test failed")
                print(f"Error: {result.stderr}")

            return dry_run_result

        except subprocess.TimeoutExpired:
            print("❌ Dry run test timed out")
            return {"success": False, "error": "timeout"}
        except Exception as e:
            print(f"❌ Dry run test error: {e}")
            return {"success": False, "error": str(e)}

    def run_actual_execution_test(self, max_runtime_minutes: int = 10) -> Dict[str, Any]:
        """Run actual metric runner execution with monitoring"""
        print(f"\n🚀 Running actual execution test (max {max_runtime_minutes} minutes)...")

        cmd = [
            sys.executable,
            str(self.project_root / "scripts" / "runner.py"),
            "--config", str(self.script_dir / "test_runs_comprehensive.cfg")
        ]

        start_time = time.time()

        try:
            # Run the metric runner
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            print("📊 Monitoring execution progress...")

            # Monitor the process
            timeout = max_runtime_minutes * 60
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return_code = -1
                print(f"⚠️  Test timed out after {max_runtime_minutes} minutes")

            end_time = time.time()
            execution_time = end_time - start_time

            execution_result = {
                "success": return_code == 0,
                "return_code": return_code,
                "execution_time": execution_time,
                "stdout": stdout,
                "stderr": stderr
            }

            if execution_result["success"]:
                print(f"✅ Execution test completed in {execution_time:.2f} seconds")
            else:
                print(f"❌ Execution test failed with return code {return_code}")
                print(f"Error output: {stderr[:500]}...")  # Show first 500 chars

            return execution_result

        except Exception as e:
            print(f"❌ Execution test error: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }

    def validate_results(self) -> Dict[str, Any]:
        """Validate test results and output files"""
        print("\n🔍 Validating test results...")

        validation_results = {
            "output_files": {},
            "log_files": {},
            "job_success_rate": 0.0,
            "device_utilization": {},
            "performance_stats": {}
        }

        # Check output files
        results_dir = self.test_root / "results"
        if results_dir.exists():
            output_files = list(results_dir.glob("*.json"))
            validation_results["output_files"]["count"] = len(output_files)
            validation_results["output_files"]["files"] = [f.name for f in output_files]

            # Validate output file contents
            valid_outputs = 0
            total_runtime = 0.0

            for output_file in output_files:
                try:
                    with open(output_file) as f:
                        data = json.load(f)

                    # Check required fields
                    required_fields = ["metric", "embedding_model", "response_count", "device", "status"]
                    if all(field in data for field in required_fields):
                        valid_outputs += 1
                        if "runtime_seconds" in data:
                            total_runtime += data["runtime_seconds"]
                except Exception as e:
                    print(f"⚠️  Invalid output file {output_file}: {e}")

            validation_results["output_files"]["valid"] = valid_outputs
            validation_results["performance_stats"]["average_job_runtime"] = (
                total_runtime / valid_outputs if valid_outputs > 0 else 0
            )

        # Check log files
        logs_dir = self.test_root / "logs"
        if logs_dir.exists():
            log_files = list(logs_dir.glob("*.log"))
            validation_results["log_files"]["count"] = len(log_files)
            validation_results["log_files"]["files"] = [f.name for f in log_files]

            # Check for both stdout and stderr logs
            stdout_logs = len([f for f in log_files if "stdout" in f.name])
            stderr_logs = len([f for f in log_files if "stderr" in f.name])
            validation_results["log_files"]["stdout_logs"] = stdout_logs
            validation_results["log_files"]["stderr_logs"] = stderr_logs

        # Calculate success rate
        total_jobs = validation_results["output_files"].get("count", 0)
        successful_jobs = validation_results["output_files"].get("valid", 0)
        validation_results["job_success_rate"] = (
            successful_jobs / total_jobs if total_jobs > 0 else 0
        )

        success_rate = validation_results["job_success_rate"]
        print(f"📊 Job success rate: {success_rate:.2%} ({successful_jobs}/{total_jobs})")
        print(f"📁 Output files: {validation_results['output_files']['count']}")
        print(f"📋 Log files: {validation_results['log_files']['count']}")

        if success_rate >= 0.9:  # 90% success rate threshold
            print("✅ Results validation passed")
        else:
            print("❌ Results validation failed - low success rate")

        return validation_results

    def _parse_device_allocations(self, stdout: str) -> List[Dict[str, str]]:
        """Parse device allocations from dry run output"""
        allocations = []
        lines = stdout.split('\n')

        for line in lines:
            if '[Device:' in line and '] STARTED' in line:
                # Extract device info from lines like "🟢 job_01: [Device: cuda:0] STARTED"
                parts = line.split('[Device: ')
                if len(parts) > 1:
                    device_part = parts[1].split(']')[0]
                    job_part = parts[0].strip()
                    allocations.append({
                        "job": job_part,
                        "device": device_part
                    })

        return allocations

    def _validate_device_allocation_logic(self, allocations: List[Dict[str, str]]):
        """Validate that device allocations follow expected patterns"""
        print("🔍 Validating device allocation logic...")

        # Check that we have proper device variety
        devices_used = [alloc["device"] for alloc in allocations]
        unique_devices = set(devices_used)

        print(f"   Devices used: {unique_devices}")

        # Check for expected device patterns
        has_single_gpu = any("cuda:" in device and "," not in device for device in devices_used)
        has_multi_gpu = any("," in device for device in devices_used)
        has_auto = any(device == "auto" for device in devices_used)

        print(f"   Single GPU jobs: {'✅' if has_single_gpu else '❌'}")
        print(f"   Multi GPU jobs: {'✅' if has_multi_gpu else '❌'}")
        print(f"   Auto (all GPU) jobs: {'✅' if has_auto else '❌'}")

        # Additional validation could check:
        # - GPU reuse patterns
        # - Priority ordering
        # - Resource conflict handling

    def run_performance_benchmark(self) -> Dict[str, Any]:
        """Run a smaller performance benchmark test"""
        print("\n⚡ Running performance benchmark...")

        # Create a minimal config for benchmarking
        benchmark_config = {
            "root_directory": str(self.test_root / "benchmark"),
            "code_directory": str(self.project_root),
            "script_path": "{{code_dir}}/scripts/segscore/tests/mock_job.py",
            "output_path_template": "{{root_dir}}/results/bench_{{metric}}_{{model_suffix}}_{{response_count}}.json",
            "logging_directory": "{{root_dir}}/logs",
            "available_gpus": 2,  # Smaller setup for benchmarking
            "dryrun": False,
            "runs": [{
                "input_path": "{{root_dir}}/data/bench_input.json",
                "metric": "benchmark",
                "embedding_model": [
                    ["bench/model-1", "mean", "bench_1"],
                    ["bench/model-2", "last", "bench_2"]
                ],
                "response_count": [100, 200],
                "min_runtime": 1.0,
                "max_runtime": 3.0,
                "failure_rate": 0.0,
                "devices": {
                    "default": {
                        "type": "cuda",
                        "count": 1
                    }
                }
            }]
        }

        # Setup benchmark environment
        benchmark_dir = self.test_root / "benchmark"
        benchmark_dir.mkdir(exist_ok=True)
        (benchmark_dir / "data").mkdir(exist_ok=True)
        (benchmark_dir / "results").mkdir(exist_ok=True)
        (benchmark_dir / "logs").mkdir(exist_ok=True)

        # Create benchmark config file
        import yaml
        benchmark_config_path = self.script_dir / "benchmark_config.cfg"
        with open(benchmark_config_path, 'w') as f:
            yaml.dump(benchmark_config, f)

        # Create benchmark data
        with open(benchmark_dir / "data" / "bench_input.json", 'w') as f:
            json.dump({"responses": [{"id": i, "text": f"bench {i}"} for i in range(500)]}, f)

        start_time = time.time()

        try:
            result = subprocess.run([
                sys.executable,
                str(self.project_root / "scripts" / "runner.py"),
                "--config", str(benchmark_config_path)
            ], capture_output=True, text=True, timeout=120)  # 2 minute timeout

            end_time = time.time()

            benchmark_result = {
                "success": result.returncode == 0,
                "execution_time": end_time - start_time,
                "jobs_completed": len(list((benchmark_dir / "results").glob("*.json"))),
                "throughput": 0.0
            }

            if benchmark_result["jobs_completed"] > 0:
                benchmark_result["throughput"] = (
                    benchmark_result["jobs_completed"] / benchmark_result["execution_time"]
                )

            print(f"⚡ Benchmark: {benchmark_result['jobs_completed']} jobs in {benchmark_result['execution_time']:.2f}s")
            print(f"⚡ Throughput: {benchmark_result['throughput']:.2f} jobs/second")

            # Cleanup
            benchmark_config_path.unlink()

            return benchmark_result

        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return {"success": False, "error": str(e)}

    def run_all_tests(self, skip_execution: bool = False, max_runtime_minutes: int = 10):
        """Run the complete test suite"""
        print("🧪 Starting Comprehensive Metric Runner Test Suite")
        print("=" * 60)

        start_time = time.time()

        # Setup
        self.setup_test_environment()

        # Test 1: Dry run validation
        dry_run_result = self.run_dry_run_test()
        self.results["test_results"]["dry_run"] = dry_run_result

        if not skip_execution:
            # Test 2: Actual execution
            execution_result = self.run_actual_execution_test(max_runtime_minutes)
            self.results["test_results"]["execution"] = execution_result

            # Test 3: Results validation
            validation_result = self.validate_results()
            self.results["test_results"]["validation"] = validation_result

            # Test 4: Performance benchmark
            benchmark_result = self.run_performance_benchmark()
            self.results["test_results"]["benchmark"] = benchmark_result

        end_time = time.time()
        total_time = end_time - start_time

        # Generate final report
        self._generate_test_report(total_time)

        return self.results

    def _generate_test_report(self, total_time: float):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("📋 FINAL TEST REPORT")
        print("=" * 60)

        print(f"⏱️  Total test time: {total_time:.2f} seconds")
        print(f"📁 Test directory: {self.test_root}")

        # Test results summary
        for test_name, result in self.results["test_results"].items():
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            print(f"{status} {test_name.replace('_', ' ').title()}")

        # Overall status
        all_passed = all(
            result.get("success", False)
            for result in self.results["test_results"].values()
        )

        print("\n" + "=" * 60)
        if all_passed:
            print("🎉 ALL TESTS PASSED!")
            print("The metric runner system is working correctly.")
        else:
            print("⚠️  SOME TESTS FAILED")
            print("Check the detailed output above for issues.")
        print("=" * 60)

        # Save detailed results
        results_file = self.test_root / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"📊 Detailed results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive metric runner tests")
    parser.add_argument("--test-root", default="/tmp/metric_runner_tests",
                       help="Root directory for test files")
    parser.add_argument("--dry-run-only", action="store_true",
                       help="Run only dry run tests (skip execution)")
    parser.add_argument("--max-runtime", type=int, default=10,
                       help="Maximum runtime for execution tests (minutes)")

    args = parser.parse_args()

    # Run the test suite
    test_suite = MetricRunnerTestSuite(args.test_root)
    results = test_suite.run_all_tests(
        skip_execution=args.dry_run_only,
        max_runtime_minutes=args.max_runtime
    )

    # Exit with appropriate code
    all_passed = all(
        result.get("success", False)
        for result in results["test_results"].values()
    )

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
