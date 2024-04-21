
from pydantic import BaseModel
from typing import Optional, List

class EvalJob(BaseModel):
    condition: str
    model_repo: str
    model_type: Optional[str] = "openchat_v3.2_mistral"
    output_repo: Optional[str] = "OpenChatDev/eval-results"
    data_path: Optional[str] = "/openchat/ochat/evaluation/eval_data"
    output_filename: str
    local_output_path: str
    eval_sets: List[str]
    system_msg: Optional[str] = ""
    tensor_parallel_size: Optional[int] = 1
    parallel: Optional[int] = 1
