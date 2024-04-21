from huggingface_hub import snapshot_download
import os
import hashlib
import filelock


def get_lock(model_name_or_path: str, cache_dir: str):
    """
    Generates a lock file for the model to prevent simultaneous downloads.

    Parameters:
    - model_name_or_path: The model name or path.
    - cache_dir: The directory for caching models.

    Returns:
    A FileLock object.
    """
    lock_dir = cache_dir
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    lock_file_name = f"{hash_name}{model_name}.lock"
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name), mode=0o666)
    return lock


def get_model(model_name_or_path: str, revision: str, cache_dir: str = os.getenv("HF_HUB_CACHE")):
    """
    Downloads or retrieves the model from cache.

    Parameters:
    - model_name_or_path: The model name or path.
    - revision: The model revision (e.g., 'main').
    - cache_dir: The directory for caching models.

    Returns:
    The folder path where the model is stored.
    """
    model_folder_path = os.path.join(cache_dir, model_name_or_path.replace("/", "-"))
    if os.path.exists(model_folder_path):
        print(f"Model {model_name_or_path} already downloaded.")
        return model_folder_path
    else:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        with get_lock(model_name_or_path, cache_dir):
            folder = snapshot_download(model_name_or_path, revision=revision, cache_dir=cache_dir)
            return folder
