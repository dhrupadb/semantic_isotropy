#!/usr/bin/env python3
"""
Test Results Validator

This script validates the output of metric runner tests to ensure
device allocation, logging, and job execution work correctly.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import argparse
from collections import defaultdict

class TestResultsValidator:
    """Validates metric runner test results"""
    
    def __init__(self, test_root: str):
        self.test_root = Path(test_root)
        self.results_dir = self.test_root / "results"
        self.logs_dir = self.test_root / "logs"
        self.validation_report = {
            "overall_status": "unknown",
            "job_analysis": {},
            "device_analysis": {},
            "timing_analysis": {},
            "error_analysis": {},
            "recommendations": []
        }
    
    def validate_job_outputs(self) -> Dict[str, Any]:
        """Validate job output files"""
        print("🔍 Validating job output files...")
        
        if not self.results_dir.exists():
            return {"error": "Results directory not found", "valid_jobs": 0, "total_jobs": 0}
        
        output_files = list(self.results_dir.glob("*.json"))
        valid_jobs = 0
        job_details = []
        device_usage = defaultdict(int)
        runtimes = []
        
        for output_file in output_files:
            try:
                with open(output_file) as f:
                    data = json.load(f)
                
                # Check required fields
                required_fields = ["metric", "embedding_model", "device", "status", "runtime_seconds"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields and data["status"] == "completed":
                    valid_jobs += 1
                    device_usage[data["device"]] += 1
                    runtimes.append(data["runtime_seconds"])
                    
                    job_details.append({
                        "file": output_file.name,
                        "metric": data["metric"],
                        "model": data["embedding_model"],
                        "device": data["device"],
                        "runtime": data["runtime_seconds"],
                        "response_count": data.get("response_count", 0),
                        "status": "valid"
                    })
                else:
                    job_details.append({
                        "file": output_file.name,
                        "status": "invalid",
                        "missing_fields": missing_fields,
                        "job_status": data.get("status", "unknown")
                    })
                    
            except Exception as e:
                job_details.append({
                    "file": output_file.name,
                    "status": "error",
                    "error": str(e)
                })
        
        analysis = {
            "total_jobs": len(output_files),
            "valid_jobs": valid_jobs,
            "success_rate": valid_jobs / len(output_files) if output_files else 0,
            "device_usage": dict(device_usage),
            "average_runtime": sum(runtimes) / len(runtimes) if runtimes else 0,
            "min_runtime": min(runtimes) if runtimes else 0,
            "max_runtime": max(runtimes) if runtimes else 0,
            "job_details": job_details
        }
        
        print(f"   📊 {valid_jobs}/{len(output_files)} jobs completed successfully")
        print(f"   ⏱️  Average runtime: {analysis['average_runtime']:.2f}s")
        print(f"   🖥️  Device usage: {dict(device_usage)}")
        
        return analysis
    
    def validate_logging(self) -> Dict[str, Any]:
        """Validate logging functionality"""
        print("📋 Validating logging functionality...")
        
        if not self.logs_dir.exists():
            return {"error": "Logs directory not found"}
        
        log_files = list(self.logs_dir.glob("*.log"))
        stdout_logs = [f for f in log_files if "stdout" in f.name]
        stderr_logs = [f for f in log_files if "stderr" in f.name]
        
        log_analysis = {
            "total_log_files": len(log_files),
            "stdout_logs": len(stdout_logs),
            "stderr_logs": len(stderr_logs),
            "log_pairs": 0,
            "errors_found": 0,
            "log_sizes": {}
        }
        
        # Check for paired logs (each job should have both stdout and stderr)
        job_ids = set()
        for log_file in log_files:
            # Extract job ID from filename like: isotropy_model_100_abc123_20240101.stdout.log
            parts = log_file.stem.split('_')
            if len(parts) >= 4:
                job_id = '_'.join(parts[:-1])  # Remove timestamp and type
                job_ids.add(job_id)
        
        log_analysis["unique_jobs_logged"] = len(job_ids)
        
        # Check log file sizes and content
        for log_file in log_files:
            size = log_file.stat().st_size
            log_analysis["log_sizes"][log_file.name] = size
            
            # Check for errors in stderr logs
            if "stderr" in log_file.name and size > 0:
                try:
                    content = log_file.read_text()
                    if any(keyword in content.lower() for keyword in ["error", "failed", "exception"]):
                        log_analysis["errors_found"] += 1
                except Exception:
                    pass
        
        print(f"   📁 {len(log_files)} log files found")
        print(f"   👥 {len(job_ids)} unique jobs logged") 
        print(f"   ⚠️  {log_analysis['errors_found']} error logs detected")
        
        return log_analysis
    
    def validate_device_allocation(self, job_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate device allocation patterns"""
        print("🖥️  Validating device allocation patterns...")
        
        device_analysis = {
            "device_types_used": set(),
            "gpu_allocation_valid": True,
            "concurrent_job_evidence": False,
            "sequential_job_evidence": False,
            "allocation_patterns": {}
        }
        
        if "job_details" not in job_analysis:
            return {"error": "No job details available for device analysis"}
        
        # Analyze device usage patterns
        device_jobs = defaultdict(list)
        for job in job_analysis["job_details"]:
            if job.get("status") == "valid":
                device = job["device"]
                device_analysis["device_types_used"].add(device.split(":")[0] if ":" in device else device)
                device_jobs[device].append(job)
        
        device_analysis["device_types_used"] = list(device_analysis["device_types_used"])
        
        # Check for proper GPU allocation
        gpu_jobs = [job for job in job_analysis["job_details"] 
                   if job.get("status") == "valid" and "cuda" in job.get("device", "")]
        
        if gpu_jobs:
            gpu_devices = [job["device"] for job in gpu_jobs]
            unique_gpu_devices = set(gpu_devices)
            
            # Check for GPU reuse (evidence of proper scheduling)
            if len(gpu_devices) > len(unique_gpu_devices):
                device_analysis["concurrent_job_evidence"] = True
                
            # Check for proper GPU numbering
            single_gpu_jobs = [d for d in gpu_devices if d.startswith("cuda:") and "," not in d]
            if single_gpu_jobs:
                gpu_numbers = [int(d.split(":")[1]) for d in single_gpu_jobs]
                expected_range = list(range(max(gpu_numbers) + 1))
                device_analysis["gpu_numbering_valid"] = set(gpu_numbers).issubset(set(expected_range))
        
        # Check for MPS sequential behavior
        mps_jobs = [job for job in job_analysis["job_details"]
                   if job.get("status") == "valid" and job.get("device") == "mps"]
        if len(mps_jobs) > 1:
            device_analysis["sequential_job_evidence"] = True
            
        device_analysis["allocation_patterns"] = {
            "total_gpu_jobs": len(gpu_jobs),
            "total_mps_jobs": len(mps_jobs),
            "total_cpu_jobs": len([j for j in job_analysis["job_details"] 
                                  if j.get("status") == "valid" and j.get("device") == "cpu"])
        }
        
        print(f"   🔧 Device types: {device_analysis['device_types_used']}")
        print(f"   🔄 GPU job scheduling: {'✅' if device_analysis.get('gpu_numbering_valid', True) else '❌'}")
        
        patterns = device_analysis.get('allocation_patterns', {})
        single_gpu = patterns.get('single_gpu_jobs', 0)
        all_gpu = patterns.get('all_gpu_jobs', 0)
        print(f"   📊 GPU allocation: {single_gpu} single-GPU, {all_gpu} all-GPU jobs")
        
        return device_analysis
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        job_analysis = self.validation_report["job_analysis"]
        device_analysis = self.validation_report["device_analysis"]
        
        # Job success rate recommendations
        success_rate = job_analysis.get("success_rate", 0)
        if success_rate < 0.9:
            recommendations.append(f"Low job success rate ({success_rate:.1%}). Check for configuration issues or resource constraints.")
        
        if success_rate < 0.7:
            recommendations.append("Very low success rate indicates potential system problems. Review error logs carefully.")
        
        # Device allocation recommendations
        if "cuda" in device_analysis.get("device_types_used", []):
            gpu_jobs = device_analysis.get("allocation_patterns", {}).get("total_gpu_jobs", 0)
            if gpu_jobs == 0:
                recommendations.append("No GPU jobs detected despite CUDA device type. Check GPU configuration.")
        
        # Logging recommendations
        log_analysis = self.validation_report.get("error_analysis", {})
        errors_found = log_analysis.get("errors_found", 0)
        if errors_found > 0:
            recommendations.append(f"Found {errors_found} error logs. Review stderr logs for details.")
        
        # Performance recommendations
        timing = self.validation_report.get("timing_analysis", {})
        avg_runtime = timing.get("average_runtime", 0)
        if avg_runtime > 30:
            recommendations.append(f"Average job runtime is high ({avg_runtime:.1f}s). Consider optimizing job parameters.")
        
        if not recommendations:
            recommendations.append("✅ All validation checks passed! The system appears to be working correctly.")
        
        return recommendations
    
    def run_validation(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        print("🧪 Starting Test Results Validation")
        print("=" * 50)
        
        # Run validation steps
        self.validation_report["job_analysis"] = self.validate_job_outputs()
        self.validation_report["error_analysis"] = self.validate_logging()
        self.validation_report["device_analysis"] = self.validate_device_allocation(
            self.validation_report["job_analysis"]
        )
        self.validation_report["timing_analysis"] = {
            "average_runtime": self.validation_report["job_analysis"].get("average_runtime", 0),
            "min_runtime": self.validation_report["job_analysis"].get("min_runtime", 0),
            "max_runtime": self.validation_report["job_analysis"].get("max_runtime", 0)
        }
        
        # Generate recommendations
        self.validation_report["recommendations"] = self.generate_recommendations()
        
        # Determine overall status
        success_rate = self.validation_report["job_analysis"].get("success_rate", 0)
        if success_rate >= 0.9:
            self.validation_report["overall_status"] = "PASS"
        elif success_rate >= 0.7:
            self.validation_report["overall_status"] = "PARTIAL"
        else:
            self.validation_report["overall_status"] = "FAIL"
        
        return self.validation_report
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 50)
        print("📋 VALIDATION SUMMARY")
        print("=" * 50)
        
        status = self.validation_report["overall_status"]
        status_emoji = {"PASS": "✅", "PARTIAL": "⚠️ ", "FAIL": "❌"}.get(status, "❓")
        print(f"{status_emoji} Overall Status: {status}")
        
        job_analysis = self.validation_report["job_analysis"]
        print(f"📊 Jobs: {job_analysis.get('valid_jobs', 0)}/{job_analysis.get('total_jobs', 0)} successful")
        
        device_analysis = self.validation_report["device_analysis"]
        device_types = device_analysis.get("device_types_used", [])
        print(f"🖥️  Device Types: {', '.join(device_types) if device_types else 'None detected'}")
        
        print("\n📋 Recommendations:")
        for i, rec in enumerate(self.validation_report["recommendations"], 1):
            print(f"   {i}. {rec}")
        
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Validate metric runner test results")
    parser.add_argument("test_root", help="Root directory of test results")
    parser.add_argument("--output", help="Output file for detailed results (JSON)")
    
    args = parser.parse_args()
    
    validator = TestResultsValidator(args.test_root)
    results = validator.run_validation()
    validator.print_summary()
    
    # Save detailed results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {args.output}")
    
    # Exit with appropriate code
    status = results["overall_status"]
    sys.exit(0 if status == "PASS" else 1 if status == "PARTIAL" else 2)


if __name__ == "__main__":
    main()