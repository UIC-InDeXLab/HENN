import numpy as np
import hnswlib
import pandas as pd
import time
import random
import sys
from scipy.spatial.distance import cdist


def load_glove(path, max_words=50000):
    word_to_index = {}
    vectors = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_words:  # limit to first N words for speed
                break
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            word_to_index[word] = i
            vectors.append(vec)

    return np.vstack(vectors), word_to_index


def evaluate_random_queries(index, glove_vectors, word_to_index, num_queries=10, k=10):
    words = list(word_to_index.keys())
    inv_index = {v: k for k, v in word_to_index.items()}

    total_recall = 0
    total_time = 0

    for i in range(num_queries):
        random.seed(i)
        word = random.choice(words)
        idx = word_to_index[word]
        query_vec = glove_vectors[idx].reshape(1, -1)

        # Get ground truth using brute-force L2
        dists = cdist(query_vec, glove_vectors, metric="euclidean")[0]
        ground_truth = np.argsort(dists)[:k]

        # Time HNSW query
        start = time.time()
        ann_labels, _ = index.knn_query(query_vec, k=k)
        total_time += time.time() - start

        # Compute recall
        ann_set = set(ann_labels[0])
        gt_set = set(ground_truth)
        recall = len(ann_set & gt_set) / k
        total_recall += recall

    avg_recall = total_recall / num_queries
    avg_time_ms = (total_time / num_queries) * 1000

    return avg_recall, avg_time_ms


glove_vectors, word_to_index = load_glove("./datasets/glove.6B.100d.txt")

print("Loaded GloVe vectors.", glove_vectors.shape)
print("Number of words:", len(word_to_index))

if __name__ == "__main__":
    method = sys.argv[1]
    # min_ef = sys.argv[2]
    # max_ef = sys.argv[3]

    dim = glove_vectors.shape[1]
    num_elements = glove_vectors.shape[0]
    print("Number of elements:", num_elements)
    print("Dimension:", dim)

    k = 10
    repeats = 6
    ids = np.arange(num_elements)

    result = {"ef": [], "recall": [], "time": []}

    for ef in [10, 20, 40, 80, 160]:
        print(f"ef: {ef}")
        for r in range(repeats):
            print("Building index...")

            # Initialize index
            p = hnswlib.Index(space="l2", dim=dim)
            p.init_index(max_elements=num_elements, ef_construction=200, M=16)

            if method == "hnsw":
                p.build_henn(glove_vectors, ids, M=4, best=False)
            elif method == "hnsw_2":
                p.set_num_threads(60)
                p.add_items(glove_vectors, ids)
            else:
                # Build HENN index
                p.build_henn(glove_vectors, ids, M=4, best=True)
            end = time.time()

            # p.save_index(f"index_glove_{method}.bin")

            print("Querying index...")

            print(f"Repeat: {r + 1}/{repeats}, ef: {ef}")

            p.set_num_threads(1)

            # Set ef (controls search accuracy vs speed)
            p.set_ef(ef)

            # Query
            recall, spent = evaluate_random_queries(
                p, glove_vectors, word_to_index, num_queries=100, k=k
            )

            print(f"Query average time: {spent:.7f}")
            print(f"Recall@10: {recall:.4f}")

            result["ef"].append(ef)
            result["recall"].append(recall)
            result["time"].append(spent)

pd.DataFrame(result).to_csv(
    f"glove_{method}.csv",
    index=False,
)

# print(pd.DataFrame(result))
