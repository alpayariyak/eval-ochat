import logging
import runpod
import asyncio

from ochat.evaluation.run_eval import run_eval

from utils.download import get_model
from utils.job import EvalJob
from utils.upload import upload_to_hf_with_retries


def handler(job_data):
    eval_job = EvalJob(job_data)
    model_path = get_model(eval_job.model_repo)
    
    try:
        asyncio.run(run_eval(
            model=model_path,
            condition=eval_job.condition,
            system_msg=eval_job.system_msg,  
            model_type=eval_job.model_type,
            eval_sets=eval_job.eval_sets,
            continue_from=None, 
            output_file=eval_job.local_output_path,
            parallel=eval_job.parallel,
            tensor_parallel_size=eval_job.tensor_parallel_size,
        ))
        logging.info("Evaluation completed successfully")
    except Exception as e:
        logging.error(f"Error running evaluation: {e}")
        return {"error": str(e)}
    
    try:
        upload_to_hf_with_retries(
            local_output_path=eval_job.local_output_path,
            output_filename=eval_job.output_filename,
            output_repo=eval_job.output_repo
        )
    except Exception as e:
        results_text = open(eval_job.local_output_path).read()
        text_eval_sets = ", ".join(eval_job.eval_sets)
        logging.error(f"Error uploading evaluation results: {e}\n###UNSAVED RESULTS FOR EVALSETS {text_eval_sets}: {results_text}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})