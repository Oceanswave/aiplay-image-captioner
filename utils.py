import os

def download_hg_model(models_dir: str, model_id: str, exDir:str=''):
    # 下载本地
    model_checkpoint = os.path.join(models_dir, exDir, os.path.basename(model_id))
    print(model_checkpoint)
    if not os.path.exists(model_checkpoint):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id, local_dir=model_checkpoint)
    return model_checkpoint