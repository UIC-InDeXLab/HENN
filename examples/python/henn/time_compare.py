import hnswlib
import numpy as np
from time import time
import sys


"""
Example of search
"""


def run_random_queries(hnsw):
    query_size = 100
    queries = np.float32(np.random.random((query_size, dim)))
    start = time()
    result, dists = hnsw.knn_query(queries, k=1)
    end = time()
    return (end - start) / query_size


def run_henn(num_elements, dim):
    data = np.float32(np.random.random((num_elements, dim)))
    henn = hnswlib.Index(space="l2", dim=dim)
    henn.init_index(
        max_elements=num_elements, ef_construction=200, M=4
    )  # HENN: m=4, HNSW : m=16
    henn.build_henn(data, np.arange(num_elements))
    henn.set_num_threads(1)
    return run_random_queries(henn)


def run_hnsw(num_elements, dim):
    data = np.float32(np.random.random((num_elements, dim)))
    hnsw = hnswlib.Index(space="l2", dim=dim)
    hnsw.init_index(
        max_elements=num_elements, ef_construction=200, M=16
    )  # HENN: m=4, HNSW : m=16
    hnsw.add_items(data, np.arange(num_elements))
    hnsw.set_num_threads(1)
    return run_random_queries(hnsw)


if __name__ == "__main__":
    repeat = 4
    dim = 32
    # logn_values = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    min_logn = sys.argv[1]
    max_logn = sys.argv[2]

    for logn in range(int(min_logn), int(max_logn) + 1):
        num_elements = 2**logn
        henn_time = 0
        hnsw_time = 0
        for i in range(repeat):
            print(f"Repeat {i + 1} / {repeat} for {logn} elements")

            if i % 2 == 0:
                tmp = run_henn(num_elements, dim)
                henn_time = tmp if tmp > henn_time else henn_time

                tmp = run_hnsw(num_elements, dim)
                hnsw_time = tmp if tmp > hnsw_time else hnsw_time
            else:
                tmp = run_hnsw(num_elements, dim)
                hnsw_time = tmp if tmp > hnsw_time else hnsw_time

                tmp = run_henn(num_elements, dim)
                henn_time = tmp if tmp > henn_time else henn_time

        print(
            f"Num elements: {num_elements}, HENN time: {henn_time:.7f}, HNSW time: {hnsw_time:.7f}"
        )
