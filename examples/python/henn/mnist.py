import hnswlib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pandas as pd
import time
import sys


def recall_at_k(pred, truth, k):
    correct = 0
    for p_row, t_row in zip(pred, truth):
        correct += len(set(p_row) & set(t_row[:k]))
    return correct / (len(pred) * k)


if __name__ == "__main__":
    method = sys.argv[1]
    ef = sys.argv[2]
    repeat = 6

    # Load MNIST
    print("Loading MNIST...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"].astype(int)
    X = X / 255.0

    # Use smaller subset for speed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1000, random_state=42
    )
    # X_train = X_train[:5000]  # reduce train size for faster brute-force baseline
    print(X_train.shape)

    dim = X_train.shape[1]
    k = 10

    result = {
        "time": [],
        "recall": [],
        "ef": [],
        "method": [],
    }

    # === Step 1: Ground-truth neighbors (exact) ===
    print("Computing exact neighbors...")
    dists = euclidean_distances(X_test, X_train)
    gt_neighbors = np.argsort(dists, axis=1)[:, :k]

    ids = np.arange(X_train.shape[0])

    for i in range(repeat):
        print(f"Repeat {i + 1}/{repeat} for method {method} with ef={ef}...")
        # === Step 2: HNSW Approximate Neighbors ===
        print("Building HNSW index...")
        p = hnswlib.Index(space="l2", dim=dim)
        p.init_index(max_elements=X_train.shape[0], ef_construction=100, M=16)

        if method == "hnsw":
            # p.add_items(X_train, ids)
            p.build_henn(X_train, ids, M=4, best=False)
        else:
            p.build_henn(X_train, ids, M=4, best=True)

        p.set_ef(int(ef))

        start = time.time()
        labels, _ = p.knn_query(X_test, k=k)
        end = time.time()
        print(f"HNSW query time: {end - start:.7f} seconds")

        # === Step 3: Compute Recall@10 ===
        recall = recall_at_k(labels, gt_neighbors, k)
        print(f"Recall@{k}: {recall:.4f}")

        result["time"].append(end - start)
        result["method"].append(method)
        result["recall"].append(recall)
        result["ef"].append(int(ef))

pd.DataFrame(result).to_csv(f"mnist_{method}_results_ef={ef}.csv", index=False)
