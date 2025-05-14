import hnswlib
import numpy as np
import pickle
import time


"""
Example of search
"""

dim = 32
num_elements = 2**16  # 10000

# Generating sample data
data = np.float32(np.random.random((num_elements, dim)))
ids = np.arange(num_elements)

# Declaring index
p = hnswlib.Index(space="l2", dim=dim)  # possible options are l2, cosine or ip

# Initializing index - the maximum number of elements should be known beforehand
p.init_index(max_elements=num_elements, ef_construction=200, M=16)

# p.set_num_threads(1)

# Element insertion (can be called several times):
start = time.time()
p.add_items(data, ids)
# p.build_henn(data, ids, M=4, best=True)
end = time.time()
print(f"Index time: {end - start:.7f}")

# p.save_index("index_synth_henn.bin")


# Controlling the recall by setting ef:
p.set_ef(50)  # ef should always be > k

# Query dataset, k - number of the closest elements (returns 2 numpy arrays)
labels, distances = p.knn_query(data, k=1)

### Index parameters are exposed as class properties:
print(
    "Recall for two batches:", np.mean(labels.reshape(-1) == np.arange(len(data))), "\n"
)

print(f"Parameters passed to constructor:  space={p.space}, dim={p.dim}")
print(f"Index construction: M={p.M}, ef_construction={p.ef_construction}")
print(f"Index size is {p.element_count} and index capacity is {p.max_elements}")
print(f"Search speed/quality trade-off parameter: ef={p.ef}")
