import numpy as np
import hnswlib
import pandas as pd
from time import time
import sys


def read_fvecs(filename):
    """Reads .fvecs file into a numpy float32 array of shape [n, d]"""
    with open(filename, "rb") as f:
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        return np.fromfile(f, dtype=np.float32).reshape(-1, d + 1)[:, 1:]


def read_ivecs(filename):
    """Reads .ivecs file into a numpy int32 array of shape [n, k]"""
    with open(filename, "rb") as f:
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.seek(0)
        return np.fromfile(f, dtype=np.int32).reshape(-1, d + 1)[:, 1:]


def recall_at_k(pred, truth, k=10):
    correct = 0
    for p_row, t_row in zip(pred, truth):
        correct += len(set(p_row) & set(t_row[:k]))
    return correct / (len(pred) * k)


def query(xq, k, p, gt):
    recalls = []
    times = []

    for i in range(len(xq)):
        query = xq[i].reshape(1, -1)

        start = time()
        pred_labels, _ = p.knn_query(query, k=k)
        elapsed = time() - start

        times.append(elapsed)
        recalls.append(len(set(pred_labels[0]) & set(gt[i][:k])) / k)

    max_time = max(times)
    avg_time = np.mean(times)
    avg_recall = np.mean(recalls)

    return max_time, avg_time, avg_recall


xb = read_fvecs("./datasets/sift/sift_base.fvecs")  # shape (1000000, 128)
xq = read_fvecs("./datasets/sift/sift_query.fvecs")  # shape (10000, 128)
gt = read_ivecs("./datasets/sift/sift_groundtruth.ivecs")  # shape (10000, 100)

print("Loaded SIFT...")
print("xb.shape", xb.shape)
print("xq.shape", xq.shape)
print("gt.shape", gt.shape)

if __name__ == "__main__":
    method = sys.argv[1]
    ef = sys.argv[2]

    dim = xb.shape[1]
    num_elements = xb.shape[0]
    k = 10
    repeats = 4
    ids = np.arange(num_elements)

    result = {
        "ef": [],
        "recall": [],
        "time": [],
        "hops": [],
        "avg_time": [],
    }

    for r in range(repeats):
        print("Building index...")

        # Initialize index
        p = hnswlib.Index(space="l2", dim=dim)
        p.init_index(max_elements=num_elements, ef_construction=200, M=16)

        if method == "hnsw":
            p.build_henn(xb, ids, M=4, best=False)
        elif method == "hnsw_2":
            p.set_num_threads(60)
            p.add_items(xb, ids)
        else:
            # Build HENN index
            p.build_henn(xb, ids, M=4, best=True)

        # p.save_index(f"index_sift_{method}.bin")

        print("Querying index...")

        for ef in [int(ef)]:
            print(f"Repeat: {r + 1}/{repeats}, ef: {ef}")

            p.set_num_threads(1)

            # Set ef (controls search accuracy vs speed)
            p.set_ef(ef)

            # Query
            start = time()
            labels, distances = p.knn_query(xq, k=k)  # shape (10000, 10)
            end = time()
            # max_time, avg_time, avg_recall = query(xq, k, p, gt)
            hops = p.get_hops()

            # print(f"Query worst time: {max_time:.7f}")
            print(f"Query average time: {end - start:.7f}")
            recall = recall_at_k(labels, gt, k=10)
            print(f"Recall@10: {recall:.4f}")
            print(f"Hops: {hops}")

            result["ef"].append(ef)
            result["recall"].append(recall)
            result["time"].append(end - start)
            result["avg_time"].append(end - start)
            result["hops"].append(hops)

# pd.DataFrame(result).to_csv(
#     f"sift_{method}_ef={ef}.csv",
#     index=False,
# )

print(result)
