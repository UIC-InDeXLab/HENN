from pathlib import Path

datasets = ["glove", "sift"]
methods = ["hnsw", "henn"]

for dataset in datasets:
    for method in methods:
        index_file = f"index_{dataset}_{method}.bin"
        file_path = Path(index_file)
        size_bytes = file_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB {dataset} {method}")