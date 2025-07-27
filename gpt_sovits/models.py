from huggingface_hub import hf_hub_download
import os

def init_model_env(env_to_files_map: dict, repo_id: str = "lj1995/GPT-SoVITS", repo_type: str = "model"):
    """
    Downloads required files for each model and sets the corresponding environment variable to the model folder.

    Example input:
        {
            "cnhubert_base_path": ("chinese-hubert-base", ("config.json", "pytorch_model.bin")),
            "bert_path": ("chinese-roberta-wwm-ext-large", ("config.json", "tokenizer.json", "pytorch_model.bin")),
        }
    """
    for env_var, (subfolder, filenames) in env_to_files_map.items():
        downloaded_paths = []
        for filename in filenames:
            print(f"Downloading {filename} for {env_var} from {subfolder}...")
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder,
                repo_type=repo_type,
            )
            downloaded_paths.append(path)

        # Use the directory of the first downloaded file to set the env var
        model_dir = os.path.dirname(downloaded_paths[0])
        os.environ[env_var] = model_dir
        print(f"Set {env_var} to {model_dir}")

def init_checkpoint(env_to_files_map: dict, repo_id: str = "lj1995/GPT-SoVITS", repo_type: str = "model"):
    for env_var, (subfolder, filename) in env_to_files_map.items():
        print(f"Downloading {filename} for {env_var} from {subfolder}...")
        if subfolder:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder,
                repo_type=repo_type,
            )
        else:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
            )
        os.environ[env_var] = path
        print(f"Set {env_var} to {path}")