from pathlib import Path
from huggingface_hub import snapshot_download

# 目标模型
repo_id = "Qwen/Qwen2.5-3B-Instruct"

# 目标保存路径
save_dir = Path(__file__).parent.parent / "pretrained" / "Qwen" / "Qwen2.5-3B-Instruct"
save_dir.mkdir(parents=True, exist_ok=True)

# 下载
snapshot_download(
    repo_id=repo_id,
    local_dir=save_dir,
    local_dir_use_symlinks=False,  # 推荐关掉，避免软链接问题
    resume_download=True
)

print(f"Model downloaded to: {save_dir}")
