import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 走国内镜像

from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="google/ddpm-celebahq-256",
    local_dir=r"D:\pythonprojects\practice-github\20261109\ddpm-celebahq-256-pretrained",
    resume_download=True,   # 断点续传，中断后重跑即可继续
)
print("下载完成，保存在：", path)
