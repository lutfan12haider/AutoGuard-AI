
from huggingface_hub import snapshot_download

dataset_path = snapshot_download(
    repo_id="rayyyyy007/kwerimoto-engine-sounds",
    repo_type="dataset",
    local_dir="data/kwerimoto"
)

print("Dataset downloaded successfully!")
print("Location:", dataset_path)
