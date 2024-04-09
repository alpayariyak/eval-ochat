import logging
import os
from dotenv import load_dotenv
import runpod
from huggingface_hub import HfApi
from download import get_model
from ochat.evaluation.run_eval import run_eval
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)


async def handler(job):
    """
    Handles a given job by running an evaluation of a model against specified datasets
    and uploading the results to a Hugging Face repository.

    Parameters:
    - job: A dictionary containing job specifications.

    Returns:
    A dictionary with the path to the output file or an error message.
    """
    job_input = job['input']
    condition = job_input['condition']
    model_repo = job_input['model_repo']
    model_type = job_input.get('model_type', "openchat_v3.2_mistral")
    output_repo = job_input.get('output_repo', "OpenChatDev/eval-results")
    output_filename = job_input['output_filename']
    eval_sets = job_input['eval_sets']
    system_msg = job_input.get('system_msg', "")

    model_path = get_model(model_repo, revision="main", cache_dir=os.getenv("HF_HUB_CACHE"))
    local_output_path = f"/results/{output_filename}"
    try:
        result = await run_eval(
            model=model_path,
            condition=condition,
            system_msg=system_msg,  
            model_type=model_type,
            data_path="/openchat/ochat/evaluation/eval_data", # "/openchat/ochat/evaluation/eval_data
            eval_sets=eval_sets,
            continue_from=None, 
            output_file=local_output_path,
            parallel=1, 
            tensor_parallel_size=1
        )
        logging.info(f"Evaluation completed successfully: {result}")
    except Exception as e:
        logging.error(f"Error running evaluation: {e}")
        return {"error": f"Error running evaluation: {e}"}
    try:
        upload_to_hf(local_output_path, output_filename, output_repo)
        return {"success": f"Results uploaded to {output_repo}"}
    except Exception as e:
        logging.error(f"Error uploading to Hugging Face: {e}")
        return {"error": f"Error uploading results to {output_repo}"}

def upload_to_hf(local_output_path, output_filename, output_repo):
    """
    Uploads the output file to a specified Hugging Face repository.

    Parameters:
    - local_output_path: The local path of the file to upload.
    - output_filename: The name of the file in the repository.
    - output_repo: The repository to upload the file to.
    """
    hf = HfApi()
    if not hf.repo_exists(repo_id=output_repo, repo_type="dataset"):
        hf.create_repo(repo_id=output_repo, repo_type="dataset")
    hf.upload_file(path_or_fileobj=local_output_path, path_in_repo=output_filename, repo_id=output_repo, repo_type="dataset")

runpod.serverless.start({"handler": handler})

