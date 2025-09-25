import json
import os
import time
import datetime
import tempfile
import logging

from semantic_isotropy.prompts.oeq import SYSTEM_PROMPT

from openai import OpenAI
from typing import List, Dict, Any, Optional


class OpenAIBatchAPI:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI Batch API client.

        Args:
            api_key: OpenAI API key. If not provided, will use OPENAI_API_KEY environment variable.
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        if not self.client.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def prepare_batch_file(self, requests: List[Dict[str, Any]], output_path: Optional[str] = None) -> str:
        """
        Prepare a batch file by creating a JSONL file with the requests.

        Args:
            requests: List of request dictionaries in the format expected by OpenAI batch API
            output_path: Path where the JSONL file should be saved

        Returns:
            Path to the created JSONL file

        Example request format:
        {
            "custom_id": "my-request-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello!"}]
            }
        }
        """
        if output_path is None:
            # Use tempfile for better security and cross-platform compatibility
            output_path = os.path.join(tempfile.gettempdir(), f"batch_requests_{int(time.time() * 1000)}.jsonl")

        try:
            with open(output_path, 'w') as f:
                for request in requests:
                    f.write(json.dumps(request) + '\n')

            self.logger.info(f"Batch file prepared successfully at {output_path} with {len(requests)} requests")
            return output_path
        except Exception as e:
            self.logger.error(f"Error preparing batch file: {e}")
            raise

    def upload_batch_file(self, file_path: str) -> Any:
        """
        Upload a batch file to OpenAI and return the batch file object.

        Args:
            file_path: Path to the JSONL file to upload

        Returns:
            Batch file object for future reference
        """
        try:
            with open(file_path, 'rb') as f:
                batch_input_file = self.client.files.create(
                    file=f,
                    purpose="batch"
                )

            self.logger.info(f"Batch file uploaded successfully. Batch file ID: {batch_input_file.id}")

            return batch_input_file

        except Exception as e:
            self.logger.error(f"Error uploading batch file: {e}")
            raise

    def create_batch_job(self, batch_input_file: Any, completion_window: str = "24h") -> Any:
        """
        Create a batch job using the uploaded batch file.

        Args:
            batch_input_file: Uploaded batch file object
            completion_window: Time window for job completion (default: "24h")

        Returns:
            Batch job object
        """
        try:
            batch_job = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window=completion_window
            )

            self.logger.info(f"Batch job using {batch_input_file.id} created successfully. Batch job ID: {batch_job.id}")

            return batch_job
        except Exception as e:
            self.logger.error(f"Error creating batch job: {e}")
            raise

    def check_batch_status(self, batch_job_id: str) -> Dict[str, Any]:
        """
        Check the status of a batch job.

        Args:
            batch_job: Batch job object

        Returns:
            Dictionary containing batch job status information
        """
        batch_job = self.client.batches.retrieve(batch_job_id)

        status_info = {
            "id": batch_job.id,
            "status": batch_job.status,
            "created_at": batch_job.created_at,
            "completed_at": batch_job.completed_at,
            "request_counts": batch_job.request_counts,
            "metadata": batch_job.metadata
        }

        self.logger.info(f"Batch job {batch_job.id} status: {batch_job.status}")
        return status_info

    def download_batch_results(self, batch_job_id: str, output_path: Optional[str] = None) -> str:
        """
        Download the results of a completed batch job.

        Args:
            batch_job: Batch job object
            output_path: Path where the results should be saved

        Returns:
            Path to the downloaded results file
        """
        if output_path is None:
            # Use tempfile for better security and cross-platform compatibility
            output_path = os.path.join(tempfile.gettempdir(), f"batch_results_{batch_job_id}_{int(time.time() * 1000)}.jsonl")

        try:
            # First check if the batch job is completed
            status_info = self.check_batch_status(batch_job_id)
            if status_info["status"] != "completed":
                raise ValueError(f"Batch job {batch_job_id} is not completed. Current status: {status_info['status']}")

            # Download the results
            batch = self.client.batches.retrieve(batch_job_id)

            requests_count = batch.request_counts

            if requests_count is not None:
                failed_count = requests_count.failed
                completed_count = requests_count.completed
                total_count = requests_count.total
                self.logger.info(f"Completed requests: {completed_count}/{total_count} ({completed_count/total_count*100:.2f}%)")
                self.logger.info(f"Failed requests: {failed_count}/{total_count} ({failed_count/total_count*100:.2f}%)")
                if failed_count > 0:
                    self.logger.warning(f"{failed_count} requests failed in batch job {batch_job.id}. Downloading error file.")
                    if batch.error_file_id:
                        error_file = self.client.files.content(batch.error_file_id)
                        error_output_path = output_path + ".err"
                        with open(error_output_path, 'wb') as ef:
                            ef.write(error_file.read())
                        self.logger.info(f"Batch error file downloaded successfully to {error_output_path}")
                    else:
                        self.logger.warning(f"No error_file_id found for batch job {batch_job.id} despite failures.")

                if completed_count > 0:
                    output_file = self.client.files.content(batch.output_file_id)
                    with open(output_path, 'wb') as f:
                        f.write(output_file.read())
                    self.logger.info(f"Batch results downloaded successfully to {output_path}")
                    return output_path
            else:
                raise ValueError(f"No requests completed in batch job {batch_job_id}. Batch: {str(batch)}.")
        except Exception as e:
            self.logger.error(f"Error downloading batch results: {e}")
            raise

    def wait_for_batch_completion(self, batch_job: Any, check_interval: int = 300) -> Dict[str, Any]:
        """
        Wait for a batch job to complete and return the final status.

        Args:
            batch_job: Batch job object
            check_interval: Interval in seconds between status checks (default: 60)

        Returns:
            Final status information of the batch job
        """
        self.logger.info(f"Waiting for batch job {batch_job.id} to complete...")
        start_time = datetime.datetime.fromtimestamp(batch_job.created_at)
        while True:
            status_info = self.check_batch_status(batch_job)

            if status_info["status"] in ["completed", "failed", "expired"]:
                self.logger.info(f"Batch job {batch_job.id} finished with status: {status_info['status']}")
                return status_info

            elapsed_time = (datetime.datetime.now() - start_time).total_seconds() // 60
            self.logger.info(f"Batch job {batch_job.id} still running after {elapsed_time:.1f} minutes. Checking again in {check_interval} seconds...")
            time.sleep(check_interval)

            # Timeout after 50 hours. Max OpenAI batch job duration is 24 hours + Buffer for finalizing.
            if elapsed_time > 50 * 60:
                self.logger.error(f"Timeout: Batch job {batch_job.id} did not complete within 50 hours.")
                raise TimeoutError(f"Batch job {batch_job.id} did not complete within 50 hours.")


# Convenience functions for common use cases
def create_chat_completion_batch(requests: List[Dict[str, Any]], model: str = "gpt-4o-mini", sampling_params: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
    """
    Create a batch file for chat completion requests.

    Args:
        requests: List of chat completion requests
        model: Model to use for the batch completion
        sampling_params: Sampling parameters to use for the batch completion
    Returns:
        List of batch request dictionaries
    """
    batch_requests = []
    sampling_params_dict = sampling_params.copy()
    if 'logprobs' in sampling_params_dict and not sampling_params_dict['logprobs']:
        sampling_params_dict.pop('logprobs')
        sampling_params_dict.pop('top_logprobs')
    if 'top_logprobs' in sampling_params_dict and sampling_params_dict['top_logprobs'] <= 1:
        sampling_params_dict.pop('top_logprobs')

    for i, request in enumerate(requests):
        batch_request = {
            "custom_id": f"chat-completion-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": model,
                     "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": request.replace(SYSTEM_PROMPT, "")}],
                     **sampling_params_dict}
        }
        batch_requests.append(batch_request)

    return batch_requests


def process_batch_completions(requests: List[Dict[str, Any]],
                            model: str = "gpt-4o-mini",
                            sampling_params: Dict[str, Any] = {},
                            temp_file_path: Optional[str] = None,
                            results_path: Optional[str] = None,
                            completion_window: str = "24h") -> str:
    """
    Complete workflow to process a batch of chat completion requests.

    Args:
        requests: List of chat completion requests
        temp_file_path: Temporary path for the batch file
        results_path: Path where results will be saved
        completion_window: Time window for job completion

    Returns:
        Path to the results file
    """

    batch_api = OpenAIBatchAPI()

    # Prepare the batch file using the convenience function
    batch_requests = create_chat_completion_batch(requests, model, sampling_params)

    batch_file_path = batch_api.prepare_batch_file(batch_requests, temp_file_path)

    # Upload the batch file
    batch_file = batch_api.upload_batch_file(batch_file_path)

    # Create the batch job
    batch_job = batch_api.create_batch_job(batch_file, completion_window)

    # Wait for completion
    batch_api.wait_for_batch_completion(batch_job)

    # Download results
    results_file_path = batch_api.download_batch_results(batch_job, results_path)

    # Clean up temporary file
    try:
        os.remove(batch_file_path)
    except OSError:
        pass

    return results_file_path, batch_job
