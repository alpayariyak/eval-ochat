from huggingface_hub import HfApi
import logging
from time import sleep

def validate_or_create_hf_repo(hf_client, repo_id):
    if not hf_client.repo_exists(repo_id=repo_id, repo_type="dataset"):
        hf_client.create_repo(repo_id=repo_id, repo_type="dataset")
        logging.info(f"Created new repository: {repo_id}")
        

def upload_to_hf_with_retries(local_output_path, output_filename, output_repo, retries=10, retry_interval=10):
    """
    Uploads the output file to a specified Hugging Face repository.

    Parameters:
    - local_output_path: The local path of the file to upload.
    - output_filename: The name of the file in the repository.
    - output_repo: The repository to upload the file to.
    """
    hf = HfApi()    
    validate_or_create_hf_repo(hf, output_repo)
    for i in range(retries):
        try:
            hf.upload_file(path_or_fileobj=local_output_path, path_in_repo=output_filename, repo_id=output_repo, repo_type="dataset")
            logging.info(f"Uploaded output file to {output_repo}")
            break
        except Exception as e:
            if i == retries - 1:
                raise e
            logging.info(f"Failed to upload file on try {i}. Retrying in {retry_interval} seconds...")
            sleep(retry_interval)