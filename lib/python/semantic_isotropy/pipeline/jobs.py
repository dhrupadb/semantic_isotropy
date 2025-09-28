import yaml
import subprocess
import concurrent.futures
import threading
from typing import List, Dict, Any, Union, Tuple, Optional
from itertools import product
from pathlib import Path
import torch
import time
import datetime
import hashlib

from semantic_isotropy.pipeline.utils import init_logger

logger = init_logger(__name__)


class JobLogManager:
    """Manages individual job logging with stdout/stderr redirection"""

    def __init__(self, logging_directory: str):
        self.logging_directory = Path(logging_directory)
        self.logging_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized job logging in: {self.logging_directory}")

    def create_job_id(self, params: Dict[str, Any]) -> str:
        """Create a unique job ID based on parameters"""
        # Create a hash of the key parameters for a unique but readable ID
        key_params = {
            'metric': params.get('metric', 'unknown'),
            'model_suffix': params.get('model_suffix', 'unknown'),
            'response_count': params.get('response_count', 0),
            'input_path': params.get('input_path', 'unknown')
        }

        # Only include pooling in hash if it's not empty/None
        pooling = params.get('pooling', '')
        if pooling and pooling.strip():
            key_params['pooling'] = pooling

        # Create readable prefix using the model_suffix
        prefix = f"{key_params['metric']}_{key_params['model_suffix']}_{key_params['response_count']}"

        # Add hash for uniqueness
        param_str = str(sorted(key_params.items()))
        hash_suffix = hashlib.md5(param_str.encode()).hexdigest()[:8]

        return f"{prefix}_{hash_suffix}"

    def get_log_files(self, job_id: str) -> Tuple[Path, Path]:
        """Get stdout and stderr log file paths for a job"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        stdout_file = self.logging_directory / f"{job_id}_{timestamp}.stdout.log"
        stderr_file = self.logging_directory / f"{job_id}_{timestamp}.stderr.log"
        return stdout_file, stderr_file

    def cleanup_old_logs(self, max_age_days: int = 7):
        """Clean up log files older than specified days"""
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        for log_file in self.logging_directory.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                logger.debug(f"Cleaned up old log file: {log_file}")


class JobProgressTracker:
    """Tracks job progress and displays real-time status"""

    def __init__(self):
        self.running_jobs = {}  # job_id -> {'start_time': time, 'params': dict, 'device': str, 'log_files': tuple}
        self.waiting_jobs = []  # List of job parameters waiting to be executed
        self.completed_jobs = []  # List of completed job results
        self.display_lock = threading.Lock()

    def add_waiting_job(self, params: Dict[str, Any]):
        """Add a job to the waiting queue"""
        with self.display_lock:
            self.waiting_jobs.append(params)

    def start_job(self, job_id: str, params: Dict[str, Any], device: str, log_files: Tuple[Path, Path]):
        """Mark a job as started"""
        with self.display_lock:
            # Remove from waiting queue if present
            self.waiting_jobs = [p for p in self.waiting_jobs if p != params]

            self.running_jobs[job_id] = {
                'start_time': time.time(),
                'params': params,
                'device': device,
                'log_files': log_files
            }
            self.update_display()

    def complete_job(self, job_id: str, result: Dict[str, Any]):
        """Mark a job as completed"""
        with self.display_lock:
            if job_id in self.running_jobs:
                job_info = self.running_jobs.pop(job_id)
                result['log_files'] = job_info['log_files']
                self.completed_jobs.append(result)
                self.update_display()

    def get_waiting_jobs_count(self) -> int:
        """Get the current count of waiting jobs"""
        with self.display_lock:
            return len(self.waiting_jobs)

    def get_next_waiting_job(self) -> Optional[Dict[str, Any]]:
        """Get the next waiting job (FIFO order)"""
        with self.display_lock:
            if self.waiting_jobs:
                return self.waiting_jobs.pop(0)
            return None

    def update_display(self):
        """Update the progress display"""
        # Clear screen and move cursor to top
        print("\033[2J\033[H", end="")

        print("=" * 80)
        print("METRIC RUNNER PROGRESS")
        print("=" * 80)

        # Show running jobs
        if self.running_jobs:
            print(f"\n🏃 RUNNING JOBS ({len(self.running_jobs)}):")
            print("-" * 80)
            for job_id, info in self.running_jobs.items():
                duration = time.time() - info['start_time']
                model = info['params'].get('embedding_model', 'unknown').split('/')[-1]
                metric = info['params'].get('metric', 'unknown')
                response_count = info['params'].get('response_count', 0)
                device = info['device']

                print(f"  {job_id[:40]:<40} | {device:<8} | {duration:>6.1f}s | {metric}/{model}/{response_count}")
        else:
            print(f"\n🏃 RUNNING JOBS (0): None")

        # Show waiting jobs queue
        if self.waiting_jobs:
            print(f"\n⏳ WAITING JOBS ({len(self.waiting_jobs)}):")
            print("-" * 80)
            for i, params in enumerate(self.waiting_jobs[:10]):  # Show first 10
                model = params.get('embedding_model', 'unknown').split('/')[-1]
                metric = params.get('metric', 'unknown')
                response_count = params.get('response_count', 0)
                print(f"  {i+1:>2}. {metric}/{model}/{response_count}")

            if len(self.waiting_jobs) > 10:
                print(f"  ... and {len(self.waiting_jobs) - 10} more jobs")
        else:
            print(f"\n⏳ WAITING JOBS (0): None")

        # Show recently completed jobs (last 5)
        if self.completed_jobs:
            recent_completed = self.completed_jobs[-5:]
            print(f"\n✅ RECENTLY COMPLETED ({len(recent_completed)}):")
            print("-" * 80)
            for result in recent_completed:
                status = "✓" if result['status'] == 'success' else "✗"
                duration = result.get('duration', 0)
                log_files = result.get('log_files', (None, None))
                stdout_log = log_files[0].name if log_files[0] else 'N/A'

                # Extract job info from command or params
                command = result.get('command', '')
                if '--embedding-model' in command:
                    parts = command.split()
                    try:
                        model_idx = parts.index('--embedding-model') + 1
                        model = parts[model_idx].split('/')[-1] if model_idx < len(parts) else 'unknown'
                    except (ValueError, IndexError):
                        model = 'unknown'
                else:
                    model = 'unknown'

                print(f"  {status} {model:<20} | {duration:>6.1f}s | Log: {stdout_log}")

        print(f"\nTotal: {len(self.completed_jobs)} completed, {len(self.running_jobs)} running, {len(self.waiting_jobs)} waiting")
        print("=" * 80)


class DeviceManager:
    """Unified device manager supporting CUDA, MPS, and CPU with optional dry-run mode"""

    def __init__(self, device_type: str, available_gpus: List[int] = None,
                 max_cpu_concurrency: int = 4, dry_run: bool = False):
        self.device_type = device_type.lower()
        self.dry_run = dry_run
        self.available_gpus = set(available_gpus) if available_gpus else (set([0, 1, 2, 3]) if dry_run else set())
        self.max_cpu_concurrency = max_cpu_concurrency
        self.allocated_gpus = set()
        self.active_cpu_jobs = 0

        # Setup threading synchronization (used by both dry-run and real modes)
        self.device_lock = threading.Lock()
        self.device_condition = threading.Condition(self.device_lock)

        mode = "DRY RUN" if dry_run else "REAL"
        logger.info(f"Initialized {mode} DeviceManager with type: {self.device_type}")
        if self.device_type == 'cuda':
            logger.info(f"Available GPUs: {sorted(list(self.available_gpus))}")
        elif self.device_type == 'cpu':
            logger.info(f"Max CPU concurrency: {self.max_cpu_concurrency}")
        elif self.device_type == 'mps':
            logger.info("MPS mode: sequential execution")

    def _can_allocate(self, requirement: Union[int, str, None]) -> bool:
        """Check if device requirement can be satisfied with current allocation"""
        handlers = {
            'cuda': self._can_allocate_cuda,
            'mps': lambda req: self.active_cpu_jobs == 0,
            'cpu': lambda req: self.active_cpu_jobs < self.max_cpu_concurrency
        }

        handler = handlers.get(self.device_type)
        if not handler:
            raise ValueError(f"Unsupported device type: {self.device_type}")
        return handler(requirement)

    def _can_allocate_cuda(self, requirement: Union[int, str, None]) -> bool:
        """CUDA-specific allocation check"""
        if requirement == 'all':
            return len(self.allocated_gpus) == 0
        elif requirement == 1:
            return len(self.available_gpus) - len(self.allocated_gpus) >= 1
        else:
            raise ValueError(f"Invalid GPU requirement: {requirement}. Only 1 or 'all' supported.")

    def acquire_device(self, requirement: Union[int, str, None] = None) -> Union[List[int], str, None]:
        """Acquire device based on requirement. Dry-run mode is non-blocking."""
        with self.device_condition:
            # In dry-run mode, don't wait - return immediately if unavailable
            if self.dry_run:
                if not self._can_allocate(requirement):
                    return None
            else:
                # In real mode, wait until requirement can be satisfied
                while not self._can_allocate(requirement):
                    wait_messages = {
                        'cuda': f"Waiting for {requirement} GPU(s) to become available...",
                        'mps': "Waiting for MPS device to become available (sequential execution)...",
                        'cpu': f"Waiting for CPU slot (active: {self.active_cpu_jobs}/{self.max_cpu_concurrency})..."
                    }
                    logger.info(wait_messages[self.device_type])
                    self.device_condition.wait()

            return self._allocate_device(requirement)

    def _allocate_device(self, requirement: Union[int, str, None]) -> Union[List[int], str]:
        """Internal method to allocate device (called after availability check)"""
        if self.device_type == 'cuda':
            return self._allocate_cuda(requirement)
        elif self.device_type == 'mps':
            self.active_cpu_jobs = 1
            if not self.dry_run:
                logger.info("Allocated MPS device")
            return 'mps'
        elif self.device_type == 'cpu':
            self.active_cpu_jobs += 1
            if not self.dry_run:
                logger.info(f"Allocated CPU slot ({self.active_cpu_jobs}/{self.max_cpu_concurrency})")
            return 'cpu'

    def _allocate_cuda(self, requirement: Union[int, str, None]) -> Union[List[int], str]:
        """CUDA-specific device allocation"""
        if requirement == 'all':
            self.allocated_gpus = self.available_gpus.copy()
            if not self.dry_run:
                logger.info(f"Allocated all {len(self.available_gpus)} GPUs")
            return 'all'
        elif requirement == 1:
            available = [gpu for gpu in sorted(self.available_gpus) if gpu not in self.allocated_gpus]
            if available:
                gpu_id = available[0]
                self.allocated_gpus.add(gpu_id)
                if not self.dry_run:
                    logger.info(f"Allocated GPU: {gpu_id}")
                return [gpu_id]
        raise ValueError(f"Invalid GPU requirement: {requirement}")

    def release_device(self, allocated_device: Union[List[int], str]):
        """Release device back to the available pool"""
        with self.device_condition:
            if self.device_type == 'cuda':
                self._release_cuda(allocated_device)
            elif self.device_type in ['mps', 'cpu']:
                if self.active_cpu_jobs > 0:
                    self.active_cpu_jobs -= 1
                    if not self.dry_run:
                        logger.info(f"Released {self.device_type.upper()} slot (active: {self.active_cpu_jobs})")

            # Notify waiting threads (only in real mode)
            if not self.dry_run:
                self.device_condition.notify_all()

    def _release_cuda(self, allocated_device: Union[List[int], str]):
        """CUDA-specific device release"""
        if allocated_device == 'all' or allocated_device == 'auto':
            self.allocated_gpus.clear()
            if not self.dry_run:
                logger.info("Released all GPUs")
        elif isinstance(allocated_device, list) and len(allocated_device) == 1:
            gpu_id = allocated_device[0]
            self.allocated_gpus.discard(gpu_id)
            if not self.dry_run:
                logger.info(f"Released GPU: {gpu_id}")
        else:
            raise ValueError(f"Invalid device to release: {allocated_device}")

    def get_status(self) -> str:
        """Get current allocation status for display"""
        if self.device_type == 'cuda':
            allocated = sorted(list(self.allocated_gpus))
            available = sorted([gpu for gpu in self.available_gpus if gpu not in self.allocated_gpus])
            return f"Allocated: {allocated}, Available: {available}"
        elif self.device_type in ['mps', 'cpu']:
            max_jobs = self.max_cpu_concurrency if self.device_type == 'cpu' else 1
            return f"Active jobs: {self.active_cpu_jobs}/{max_jobs}"
        return "Unknown"


class MetricRunner:
    """Main class for running metric generation with different parameter combinations"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.device_manager = None  # Legacy - will be deprecated
        self.device_managers_cache = {}  # Cache for different device managers
        self.log_manager = None
        self.progress_tracker = JobProgressTracker()
        self._setup_device_manager()  # Keep for backwards compatibility
        self._setup_log_manager()

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config file {self.config_path}: {e}")
            raise

    def _setup_device_manager(self):
        """Initialize device manager based on the first run's device configuration"""
        # Get device configuration from the first run (assumes all runs use same device type)
        runs = self.config.get('runs', [])
        if not runs:
            logger.error("No 'runs' section found in config")
            raise ValueError("No runs configuration found")

        first_run = runs[0]
        device_config = first_run.get('devices', {})
        default_config = device_config.get('default', {'type': 'cuda', 'count': 1, 'max_concurrency': 4})

        device_type = default_config.get('type', 'cuda').lower()
        max_concurrency = default_config.get('max_concurrency', 4)

        if device_type == 'cuda':
            # Setup CUDA device manager
            # Use configured GPU count, fallback to hardware detection, then default
            configured_gpu_count = self.config.get('available_gpus')

            if configured_gpu_count is not None:
                # Use user-configured GPU count
                available_gpus = list(range(configured_gpu_count))
                self.device_manager = DeviceManager('cuda', available_gpus)
                logger.info(f"Initialized CUDA device manager with {configured_gpu_count} configured GPUs: {available_gpus}")
            elif torch.cuda.is_available():
                # Auto-detect available GPUs
                gpu_count = torch.cuda.device_count()
                available_gpus = list(range(gpu_count))
                self.device_manager = DeviceManager('cuda', available_gpus)
                logger.info(f"Initialized CUDA device manager with {gpu_count} detected GPUs: {available_gpus}")
        elif device_type == 'mps':
            # Setup MPS device manager (Apple Silicon)
            self.device_manager = DeviceManager('mps')
            logger.info("Initialized MPS device manager for Apple Silicon")
        elif device_type == 'cpu':
            # Setup CPU device manager
            self.device_manager = DeviceManager('cpu', max_cpu_concurrency=max_concurrency)
            logger.info(f"Initialized CPU device manager with max concurrency: {max_concurrency}")
        else:
            raise ValueError(f"Unsupported device type: {device_type}")

    def _setup_log_manager(self):
        """Initialize log manager with logging directory from config"""
        logging_dir_template = self.config.get('logging_directory', './logs/metric_runner')
        logging_dir = self._substitute_templates(logging_dir_template)
        self.log_manager = JobLogManager(logging_dir)
        # Clean up old logs
        self.log_manager.cleanup_old_logs()

    def _substitute_templates(self, path: str) -> str:
        """Substitute template variables in path"""
        result = path

        # Root directory substitution
        root_directory = self.config.get('root_directory', '')
        if root_directory and '{{root_dir}}' in result:
            result = result.replace('{{root_dir}}', root_directory)

        # Code directory substitution
        code_directory = self.config.get('code_directory', '')
        if code_directory and '{{code_dir}}' in result:
            result = result.replace('{{code_dir}}', code_directory)

        return result

    def _format_output_path(self, input_path: str, params: Dict[str, Any]) -> str:
        """Generate output path based on input path and parameters with template substitution"""
        # Apply template substitution to input path
        resolved_input_path = self._substitute_templates(input_path)
        input_file = Path(resolved_input_path)
        base_name = input_file.stem

        # Check if custom output path template is specified in params (from run block)
        output_template = params.get('output_path_template')
        if output_template:
            # Apply template substitution to the output template first
            output_template = self._substitute_templates(output_template)

            # Extract parameters for template
            metric = params.get('metric', 'unknown')
            model_suffix = params.get('model_suffix', 'unknown')
            response_count = params.get('response_count', 0)
            response_max = params.get('response_max', 1000)

            # Apply parameter substitutions
            output_name = output_template
            output_name = output_name.replace('{{base_name}}', base_name)
            output_name = output_name.replace('{{metric}}', str(metric))
            output_name = output_name.replace('{{response_max}}', str(response_max))
            output_name = output_name.replace('{{response_count}}', str(response_count))
            output_name = output_name.replace('{{model_suffix}}', model_suffix)

            # Use same directory as input file
            return str(input_file.parent / output_name)
        else:
            # Fallback to default format for backward compatibility
            metric = params.get('metric', 'unknown')
            model_suffix = params.get('model_suffix', 'unknown')
            response_count = params.get('response_count', 0)

            param_parts = []
            excluded_params = {
                'input_path', 'output_path', 'script_path', 'output_dir',
                'overwrite', 'dryrun', '_device_config', '_num_devices_config', '_devices_config', 'model_suffix'
            }

            for key, value in params.items():
                if key not in excluded_params:
                    param_parts.append(f"{key}_{value}")

            param_suffix = "_".join(param_parts)
            output_name = f"{base_name}_{param_suffix}.pkl"

            return str(input_file.parent / output_name)

    def _get_gpu_requirement(self, params: Dict[str, Any]) -> Union[int, str]:
        """Determine GPU requirement from parameters"""
        device = params.get('device', 'auto')

        if device == 'auto':
            return 'all'
        elif device == 'cuda':
            return 1  # Default to single GPU
        elif isinstance(device, str) and device.startswith('cuda:'):
            # Support for multiple GPUs specified as 'cuda:0', 'cuda:1,2', or 'cuda:0,1,2'
            gpu_list = device[5:]  # Remove 'cuda:' prefix
            if gpu_list == '':
                return 1  # Just 'cuda:' means 1 GPU (default)
            # Split by comma and count GPUs
            gpu_ids = [int(g.strip().replace("cuda:", "")) for g in gpu_list.split(',') if g.strip() != '']
            return len(gpu_ids)
        elif isinstance(device, int):
            return device  # Specific number of GPUs
        else:
            # Fallback to single GPU for any CUDA device
            return 1

    def _process_model_tuple(self, model_spec: Union[str, List]) -> Tuple[str, Optional[str], str]:
        """Process model specification - handle legacy string format and new 3-element tuple format"""
        if isinstance(model_spec, list) and len(model_spec) >= 3:
            # New format: [model_name, embedding_type, model_suffix]
            model_name, embedding_type, model_suffix = model_spec[0], model_spec[1], model_spec[2]
            return model_name, embedding_type, model_suffix
        elif isinstance(model_spec, list) and len(model_spec) == 2:
            # Legacy 2-element format: [model_name, embedding_type]
            model_name, embedding_type = model_spec[0], model_spec[1]
            # Generate model suffix from model name
            model_suffix = model_name.split('/')[-1].replace('-', '_') if '/' in model_name else model_name.replace('-', '_')
            return model_name, embedding_type, model_suffix
        elif isinstance(model_spec, list) and len(model_spec) == 1:
            # Single item list - use default pooling
            model_name = model_spec[0]
            model_suffix = model_name.split('/')[-1].replace('-', '_') if '/' in model_name else model_name.replace('-', '_')
            return model_name, "mean", model_suffix
        else:
            # Legacy string format
            model_name = str(model_spec)
            model_suffix = model_name.split('/')[-1].replace('-', '_') if '/' in model_name else model_name.replace('-', '_')
            return model_name, "mean", model_suffix

    def _is_api_model(self, model_name: str) -> bool:
        """Check if model is an API-only model (OpenAI, Gemini)"""
        api_indicators = [
            'text-embedding-',  # OpenAI models
            'models/text-embedding',  # Gemini models
            'openai',
            'gemini',
            'cohere'
        ]
        return any(indicator in model_name.lower() for indicator in api_indicators)

    def _generate_parameter_combinations(self) -> List[Tuple[Dict[str, Any], int]]:
        """Generate all combinations of parameters from config with priorities"""
        runs = self.config.get('runs', [])
        if not runs:
            logger.error("No 'runs' section found in config")
            return []

        combinations = []

        for run in runs:
            # Extract device configuration (don't include in parameter combinations)
            devices_config = run.get('devices', run.get('num_devices', run.get('device', {'type': 'cuda', 'count': 1, 'max_concurrency': 4})))

            # Get parameter lists for this run
            param_lists = {}
            for param, values in run.items():
                # Skip device/num_devices/devices configuration
                if param in ['device', 'num_devices', 'devices']:
                    continue

                # Handle embedding_model specially (model tuples)
                elif param == 'embedding_model':
                    if isinstance(values, list):
                        # Process each model tuple
                        processed_models = []
                        for model_spec in values:
                            model_name, embedding_type, model_suffix = self._process_model_tuple(model_spec)
                            processed_models.append((model_name, embedding_type, model_suffix))
                        param_lists[param] = processed_models
                    else:
                        # Single model specification
                        model_name, embedding_type, model_suffix = self._process_model_tuple(values)
                        param_lists[param] = [(model_name, embedding_type, model_suffix)]

                # Handle all other parameters normally
                else:
                    # Apply root directory substitution for path parameters
                    if isinstance(values, list):
                        processed_values = []
                        for v in values:
                            if isinstance(v, str):
                                processed_values.append(self._substitute_templates(v))
                            else:
                                processed_values.append(v)
                        param_lists[param] = processed_values
                    else:
                        if isinstance(values, str):
                            param_lists[param] = [self._substitute_templates(values)]
                        else:
                            param_lists[param] = [values]  # Convert single value to list

            # Generate all combinations for this run
            param_names = list(param_lists.keys())
            param_values = [param_lists[name] for name in param_names]

            logger.info(f"Generating combinations for run with {len(param_names)} parameters:")
            for name, values in zip(param_names, param_values):
                logger.info(f"  {name}: {len(values)} values")

            for combination in product(*param_values):
                param_dict = dict(zip(param_names, combination))

                # Process embedding_model tuple into separate parameters
                if 'embedding_model' in param_dict:
                    model_name, embedding_type, model_suffix = param_dict['embedding_model']
                    param_dict['embedding_model'] = model_name
                    param_dict['model_suffix'] = model_suffix

                    # Only add pooling if it's not an API model and embedding_type is not None
                    if embedding_type is not None and not self._is_api_model(model_name):
                        param_dict['pooling'] = embedding_type

                # Determine priority based on device requirement for this specific combination
                device_config = self._get_device_config_from_params(param_dict, devices_config)
                device_type = device_config.get('type', 'cuda')
                device_count = device_config.get('count', 1)

                if device_type == 'cuda':
                    if device_count == 'all':
                        priority = 2  # Lower priority (all-GPU jobs run after single-GPU jobs)
                    elif device_count == 1:
                        priority = 1  # Higher priority (single-GPU jobs run first)
                    else:
                        raise ValueError(f"Invalid CUDA device count: {device_count}. Only 1 or 'all' supported.")
                else:
                    # For MPS and CPU, use high priority (they don't compete with GPUs)
                    priority = 1

                # Store the device config with the parameters for later use
                param_dict['_devices_config'] = devices_config

                combinations.append((param_dict, priority))

        # Sort by priority (lower number = higher priority)
        combinations.sort(key=lambda x: x[1])

        logger.info(f"Generated {len(combinations)} total parameter combinations")
        logger.info(f"Priority distribution: "
                   f"High (1-GPU, MPS, CPU): {sum(1 for _, p in combinations if p == 1)}, "
                   f"Low (all-GPU): {sum(1 for _, p in combinations if p == 2)}")

        return combinations

    def _get_device_config_from_params(self, params: Dict[str, Any], devices_config: Union[int, str, Dict[str, Any]]) -> Dict[str, Any]:
        """Determine device configuration from parameters and devices configuration"""

        # Handle legacy format (simple integer or string for GPU count)
        if isinstance(devices_config, (int, str)):
            return {'type': 'cuda', 'count': devices_config, 'max_concurrency': 4}

        # Handle new device configuration format
        if isinstance(devices_config, dict):
            default_config = devices_config.get('default', {'type': 'cuda', 'count': 1, 'max_concurrency': 4})
            overrides = devices_config.get('overrides', [])

            # Check overrides in order - first match wins
            for override in overrides:
                condition = override.get('condition', {})
                value = override.get('value', {})

                if value and self._matches_condition(params, condition):
                    logger.debug(f"Device override matched: {condition} -> {value}")
                    # Merge override with defaults
                    result = default_config.copy()
                    result.update(value)
                    return result

            # No overrides matched, use default
            logger.debug(f"Using default device config: {default_config}")
            return default_config

        # Fallback
        return {'type': 'cuda', 'count': 1, 'max_concurrency': 4}

    def _get_device_manager_for_type(self, device_type: str, device_config: Dict[str, Any]):
        """Get or create appropriate device manager for the device type"""
        # Create cache key based on device type and key parameters
        cache_key = f"{device_type}_{device_config.get('max_concurrency', 4)}"

        if cache_key in self.device_managers_cache:
            return self.device_managers_cache[cache_key]

        # Check if we should use dry-run mode - look for any run with dryrun=True
        dry_run = self.config.get('dryrun', False) or any(
            run.get('dryrun', False) for run in self.config.get('runs', [])
        )

        # Create device manager based on type
        device_type = device_type.lower()
        if device_type == 'cuda':
            # Use configured GPU count from config
            configured_gpu_count = self.config.get('available_gpus')
            available_gpus = list(range(configured_gpu_count)) if configured_gpu_count is not None else None
            manager = DeviceManager('cuda', available_gpus, dry_run=dry_run)

        elif device_type == 'mps':
            manager = DeviceManager('mps', dry_run=dry_run)

        elif device_type == 'cpu':
            max_concurrency = device_config.get('max_concurrency', 4)
            manager = DeviceManager('cpu', max_cpu_concurrency=max_concurrency, dry_run=dry_run)

        else:
            # Unknown device type, default to CUDA
            logger.warning(f"Unknown device type '{device_type}', defaulting to CUDA")
            manager = DeviceManager('cuda', dry_run=dry_run)

        # Cache the manager
        self.device_managers_cache[cache_key] = manager
        logger.debug(f"Created device manager for {device_type}: {cache_key}")

        return manager

    def _format_device_parameter(self, allocated_device: Union[List[int], str], device_type: str) -> str:
        """Format allocated device into device parameter for command line"""
        if device_type == 'cuda':
            if allocated_device == 'all' or allocated_device == 'auto':
                return 'auto'
            elif isinstance(allocated_device, list) and len(allocated_device) == 1:
                return f'cuda:{allocated_device[0]}'
            elif isinstance(allocated_device, str) and allocated_device.startswith('cuda:'):
                return allocated_device
            else:
                return str(allocated_device)
        elif device_type in ['mps', 'cpu']:
            return device_type
        else:
            return str(allocated_device)

    def _matches_condition(self, params: Dict[str, Any], condition: Dict[str, Any]) -> bool:
        """Check if parameters match the condition for device override.

        Supports both single values and lists in conditions.
        If a condition value is a list, the parameter must be contained in that list.
        If a condition value is a single value, exact match is required.
        """
        for key, expected_value in condition.items():
            if key not in params:
                return False

            param_value = params[key]

            # If expected_value is a list, check if param_value is in the list
            if isinstance(expected_value, list):
                # Handle different types within the list
                matched = False
                for expected_item in expected_value:
                    if isinstance(expected_item, (int, float)) and isinstance(param_value, (int, float)):
                        if param_value == expected_item:
                            matched = True
                            break
                    elif str(param_value) == str(expected_item):
                        matched = True
                        break

                if not matched:
                    return False
            else:
                # Single value matching (original behavior)
                # Handle numeric comparisons
                if isinstance(expected_value, (int, float)) and isinstance(param_value, (int, float)):
                    if param_value != expected_value:
                        return False
                # Handle string comparisons
                elif str(param_value) != str(expected_value):
                    return False

        return True

    def _build_command(self, params: Dict[str, Any]) -> List[str]:
        """Build command line arguments for gen_metric.py"""
        script_path_template = self.config.get('script_path', 'scripts/segscore/gen_metric.py')
        script_path = self._substitute_templates(script_path_template)

        cmd = ['python', script_path]

        # Add required parameters
        input_path = params.get('input_path')
        if not input_path:
            raise ValueError("input_path is required in parameters")

        output_path = self._format_output_path(input_path, params)

        cmd.extend(['--input-path', input_path])
        cmd.extend(['--output-path', output_path])

        # Parameters to exclude from command line (internal config parameters)
        excluded_params = {
            'input_path', 'output_path', 'script_path', 'output_dir', 'output_path_template',
            '_device_config', '_num_devices_config', '_devices_config', 'model_suffix'
        }

        # Add other parameters
        for key, value in params.items():
            if key in excluded_params:
                continue  # Skip internal config parameters

            # Skip pooling parameter if it's empty or whitespace-only
            if key == 'pooling' and (not value or not str(value).strip()):
                continue

            # Convert parameter name to command line format
            cmd_arg = f"--{key.replace('_', '-')}"

            # Handle boolean flags
            if isinstance(value, bool):
                if value:
                    cmd.append(cmd_arg)
            else:
                cmd.extend([cmd_arg, str(value)])

        return cmd

    def _run_single_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single command with device management and logging"""
        # Extract device config and remove it from params before building command
        devices_config = params.pop('_devices_config', {'type': 'cuda', 'count': 1, 'max_concurrency': 4})
        device_config = self._get_device_config_from_params(params, devices_config)
        allocated_device = None
        job_id = None
        log_files = (None, None)

        try:
            device_type = device_config.get('type', 'cuda')
            device_count = device_config.get('count', 1)

            # Create job ID and log files
            job_id = self.log_manager.create_job_id(params)
            log_files = self.log_manager.get_log_files(job_id)

            # Get appropriate device manager for this job's device type
            job_device_manager = self._get_device_manager_for_type(device_type, device_config)

            # Acquire device based on type and requirement (this will block until available)
            if device_type == 'cuda':
                allocated_device = job_device_manager.acquire_device(device_count)
            else:
                # For MPS and CPU, no specific requirement needed
                allocated_device = job_device_manager.acquire_device()

            # Update device parameter based on allocation
            params = params.copy()
            params['device'] = self._format_device_parameter(allocated_device, device_type)

            # Format device info for display
            device_display = f"{device_type.upper()}"
            if device_type == 'cuda':
                if allocated_device == 'all' or allocated_device == 'auto':
                    device_display += ":all"
                elif isinstance(allocated_device, list) and len(allocated_device) == 1:
                    device_display += f":{allocated_device[0]}"
                elif isinstance(allocated_device, str) and allocated_device.startswith('cuda:'):
                    device_display += f":{allocated_device.split(':')[1]}"

            # Update progress tracker
            self.progress_tracker.start_job(job_id, params, device_display, log_files)

            cmd = self._build_command(params)

            start_time = time.time()

            # Run with stdout/stderr redirection to log files
            with open(log_files[0], 'w') as stdout_file, open(log_files[1], 'w') as stderr_file:
                result = subprocess.run(
                    cmd,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                    check=True
                )

            end_time = time.time()

            # Read log files for result (optional, for backward compatibility)
            stdout_content = log_files[0].read_text() if log_files[0].exists() else ""
            stderr_content = log_files[1].read_text() if log_files[1].exists() else ""

            result_dict = {
                'status': 'success',
                'params': params,
                'command': ' '.join(cmd),
                'stdout': stdout_content,
                'stderr': stderr_content,
                'duration': end_time - start_time,
                'allocated_device': allocated_device,
                'device_type': device_type,
                'job_id': job_id,
                'log_files': log_files
            }

            # Update progress tracker with completion
            self.progress_tracker.complete_job(job_id, result_dict)

            return result_dict

        except subprocess.CalledProcessError as e:
            # Read log files for error info
            stdout_content = log_files[0].read_text() if log_files[0] and log_files[0].exists() else ""
            stderr_content = log_files[1].read_text() if log_files[1] and log_files[1].exists() else ""

            result_dict = {
                'status': 'failed',
                'params': params,
                'command': ' '.join(cmd) if 'cmd' in locals() else 'N/A',
                'error': str(e),
                'stdout': stdout_content,
                'stderr': stderr_content,
                'allocated_device': allocated_device,
                'device_type': device_type if 'device_type' in locals() else 'unknown',
                'job_id': job_id,
                'log_files': log_files
            }

            if job_id:
                self.progress_tracker.complete_job(job_id, result_dict)

            return result_dict

        except Exception as e:
            result_dict = {
                'status': 'error',
                'params': params,
                'error': str(e),
                'allocated_device': allocated_device,
                'device_type': device_type if 'device_type' in locals() else 'unknown',
                'job_id': job_id,
                'log_files': log_files
            }

            if job_id:
                self.progress_tracker.complete_job(job_id, result_dict)

            return result_dict

        finally:
            # Release device
            if allocated_device is not None and 'job_device_manager' in locals():
                job_device_manager.release_device(allocated_device)

    def run_all(self, dryrun: bool = None, quiet: bool = False) -> List[Dict[str, Any]]:
        """Run all parameter combinations with prioritization"""
        combinations_with_priority = self._generate_parameter_combinations()

        if not combinations_with_priority:
            logger.error("No parameter combinations to run")
            return []

        # Check dryrun setting from run blocks or parameter - if any run has dryrun enabled, use dryrun mode
        runs_dryrun = any(combination.get('dryrun', False) for combination, _ in combinations_with_priority)
        should_dryrun = dryrun if dryrun is not None else runs_dryrun

        # Extract just the parameter combinations (priority already used for sorting)
        combinations = [params for params, _ in combinations_with_priority]

        # Populate waiting queue for progress display
        for params in combinations:
            self.progress_tracker.add_waiting_job(params.copy())

        if should_dryrun:
            if not quiet:
                print("Dry run mode - simulating job execution with device allocation:")
                print("=" * 100)

                # Get the primary device type for display (from first job)
                primary_device_type = 'cuda'  # Default
                if combinations:
                    first_params = combinations[0].copy()
                    first_devices_config = first_params.pop('_devices_config', {'type': 'cuda', 'count': 1})
                    first_device_config = self._get_device_config_from_params(first_params, first_devices_config)
                    primary_device_type = first_device_config.get('type', 'cuda')

                print(f"Device Manager: {primary_device_type.upper()}")
                print(f"Available Resources: Simulating device allocation for mixed device types")
                print()

            # Create cache of dry run device managers for different device types
            dry_run_managers = {}
            running_jobs = {}  # job_id -> (device_manager, device, params)
            waiting_queue = []  # List of (params, device_config) waiting for resources
            completed_count = 0

            # Process all jobs through simulation
            for job_idx, params in enumerate(combinations):
                params_copy = params.copy()
                devices_config = params_copy.pop('_devices_config', {'type': 'cuda', 'count': 1})
                device_config = self._get_device_config_from_params(params_copy, devices_config)
                device_type = device_config.get('type', 'cuda')
                device_count = device_config.get('count', 1)

                # Get or create dry run device manager for this device type
                manager_key = f"{device_type}_{device_config.get('max_concurrency', 4)}"
                if manager_key not in dry_run_managers:
                    dry_run_managers[manager_key] = self._get_device_manager_for_type(device_type, device_config)

                dry_run_manager = dry_run_managers[manager_key]

                # Apply template substitution to params before building command
                for key, value in params_copy.items():
                    if isinstance(value, str) and ('{{' in value and '}}' in value):
                        params_copy[key] = self._substitute_templates(value)

                job_id = f"job_{job_idx+1:02d}"

                # Try to allocate device
                requirement = device_count if device_type == 'cuda' else None
                allocated_device = dry_run_manager.acquire_device(requirement)

                if allocated_device:
                    # Job can start immediately - set device parameter
                    params_copy['device'] = self._format_device_parameter(allocated_device, device_type)
                    cmd = self._build_command(params_copy)
                    running_jobs[job_id] = (dry_run_manager, allocated_device, params_copy)

                    # Display the device in user-friendly format
                    display_device = params_copy['device']  # Use the formatted device string
                    if quiet:
                        print(' '.join(cmd))
                    else:
                        print(f"🟢 {job_id}: [Device: {display_device}] STARTED")
                        print(f"    Command: {' '.join(cmd)}")
                        print(f"    Resources: {dry_run_manager.get_status()}")
                        print()

                    # Simulate job completion immediately for dry run
                    dry_run_manager.release_device(allocated_device)
                    del running_jobs[job_id]
                    completed_count += 1

                    if not quiet:
                        print(f"✅ {job_id}: COMPLETED, resources released")
                        print(f"    Resources: {dry_run_manager.get_status()}")
                        print()

                    # Try to start any waiting jobs
                    new_waiting = []
                    for waiting_params, waiting_device_config, waiting_manager_key in waiting_queue:
                        waiting_device_type = waiting_device_config.get('type', 'cuda')
                        waiting_device_count = waiting_device_config.get('count', 1)
                        waiting_manager = dry_run_managers[waiting_manager_key]

                        waiting_requirement = waiting_device_count if waiting_device_type == 'cuda' else None
                        waiting_device = waiting_manager.acquire_device(waiting_requirement)

                        if waiting_device:
                            # Waiting job can now start - set device parameter
                            waiting_params['device'] = self._format_device_parameter(waiting_device, waiting_device_type)

                            waiting_cmd = self._build_command(waiting_params)
                            waiting_job_id = f"job_{len(combinations) - len(waiting_queue) + len(new_waiting) + 1:02d}"

                            display_waiting_device = waiting_params['device']
                            if quiet:
                                print(' '.join(waiting_cmd))
                            else:
                                print(f"🟡 {waiting_job_id}: [Device: {display_waiting_device}] STARTED (from queue)")
                                print(f"    Command: {' '.join(waiting_cmd)}")
                                print(f"    Resources: {waiting_manager.get_status()}")
                                print()

                            # Simulate completion
                            waiting_manager.release_device(waiting_device)
                            completed_count += 1

                            if not quiet:
                                print(f"✅ {waiting_job_id}: COMPLETED")
                                print(f"    Resources: {waiting_manager.get_status()}")
                                print()
                        else:
                            new_waiting.append((waiting_params, waiting_device_config, waiting_manager_key))

                    waiting_queue = new_waiting

                else:
                    # Job must wait for resources
                    if not quiet:
                        print(f"🔴 {job_id}: WAITING for resources (need {device_count if device_count != 1 else 'device'})")
                        print(f"    Resources: {dry_run_manager.get_status()}")
                        print()
                    waiting_queue.append((params_copy, device_config, manager_key))

            if not quiet:
                print("=" * 100)
                print(f"DRY RUN SUMMARY:")
                print(f"Total jobs: {len(combinations)}")
                print(f"Completed: {completed_count}")
                print(f"Still waiting: {len(waiting_queue)}")
                if waiting_queue:
                    print(f"Waiting jobs would require additional resources or sequential processing")
                print("=" * 100)

            return []

        # Determine execution strategy based on device type
        device_type = self.device_manager.device_type

        # Initialize progress display
        self.progress_tracker.update_display()

        # Use the new job queue system for all device types
        # This ensures proper resource management and waiting job updates
        results = self._run_with_job_queue(combinations)

        # Final summary
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = len(results) - successful

        # Clear screen one final time and show completion summary
        print("\033[2J\033[H", end="")
        print("=" * 80)
        print("METRIC RUNNER COMPLETED")
        print("=" * 80)
        print(f"\nTotal: {len(results)} jobs")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

        if failed > 0:
            print(f"\n❌ FAILED JOBS:")
            print("-" * 80)
            for result in results:
                if result['status'] != 'success':
                    log_files = result.get('log_files', (None, None))
                    stderr_log = log_files[1].name if log_files[1] else 'N/A'
                    job_id = result.get('job_id', 'unknown')
                    print(f"  {job_id} | Error log: {stderr_log}")

        if successful > 0:
            print(f"\n✅ ALL JOB LOGS LOCATION:")
            print(f"  {self.log_manager.logging_directory}")

        print("=" * 80)

        return results

    def _run_with_job_queue(self, combinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run jobs using a proper queue system that respects resource availability"""
        results = []
        active_futures = {}  # future -> params
        max_concurrent_jobs = self._get_max_concurrent_jobs()

        # Create thread pool with appropriate size
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_jobs) as executor:

            # Submit initial batch of jobs that can run concurrently
            # Note: combinations are already added to progress_tracker.waiting_jobs by run_all()
            while len(active_futures) < max_concurrent_jobs and self.progress_tracker.get_waiting_jobs_count() > 0:
                next_params = self.progress_tracker.get_next_waiting_job()
                if next_params:
                    future = executor.submit(self._run_single_command, next_params)
                    active_futures[future] = next_params

            last_update = 0
            update_interval = 1.0  # seconds

            while active_futures or self.progress_tracker.get_waiting_jobs_count() > 0:
                # Update display every second (or two)
                now = time.time()
                if now - last_update >= update_interval:
                    self.progress_tracker.update_display()
                    last_update = now

                # Wait for any job to complete, but don't block too long so we can update display
                if active_futures:
                    # Use as_completed with a short timeout to allow periodic display updates
                    done, not_done = concurrent.futures.wait(
                        active_futures.keys(), timeout=0.5, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    if done:
                        completed_future = next(iter(done))
                        # Get the result and remove from active futures
                        params = active_futures.pop(completed_future)
                        result = completed_future.result()
                        results.append(result)
                else:
                    # If no active futures, just sleep a bit to avoid busy loop
                    time.sleep(0.2)

                # Try to submit next waiting job if any remain and we have capacity
                while len(active_futures) < max_concurrent_jobs:
                    next_params = self.progress_tracker.get_next_waiting_job()
                    if next_params:
                        next_future = executor.submit(self._run_single_command, next_params)
                        active_futures[next_future] = next_params
                    else:
                        break
            # One last update at the end
            self.progress_tracker.update_display()
        return results

    def _get_max_concurrent_jobs(self) -> int:
        """Determine maximum number of concurrent jobs based on device type"""
        device_type = self.device_manager.device_type

        if device_type == 'cuda':
            return len(self.device_manager.available_gpus)
        elif device_type == 'cpu':
            return self.device_manager.max_cpu_concurrency
        elif device_type == 'mps':
            return 1  # MPS is sequential
        else:
            return 4  # Default fallback
