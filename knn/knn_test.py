import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer

# Verification with sklearn
VERIFY = True

# Load the CUDA shared library
lib = ctypes.CDLL('./knn.so')

lib.cuda_knn.argtypes = [
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]
lib.cuda_knn.restype = None

def knn_cuda(data, query, k, threads=256):
    n_points, dim = data.shape
    
    data = np.ascontiguousarray(data, dtype=np.float32)
    query = np.ascontiguousarray(query, dtype=np.float32)
    
    indices = np.zeros(k, dtype=np.int32)
    distances = np.zeros(k, dtype=np.float32)
    
    lib.cuda_knn(data, query, indices, distances, n_points, dim, k, threads)
    
    return indices, distances

if __name__ == "__main__":
    np.random.seed(42)
    n_points = 1000
    dim = 10
    k = 5
    
    data = np.random.rand(n_points, dim).astype(np.float32)
    query = np.random.rand(dim).astype(np.float32)
    
    # testing various thread counts
    thread_counts = [128, 256, 512, 1024]
    for threads in thread_counts:
        print(f"\nTesting with {threads} threads:")
        indices, distances = knn_cuda(data, query, k, threads=threads)
        
        print("Nearest neighbor indices:", indices)
        print("Distances:", distances)
    
    # double checking with sklearn
    if VERIFY:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(data)
        sk_distances, sk_indices = nbrs.kneighbors([query])
        print("\nSklearn verification:")
        print("Indices:", sk_indices[0])
        print("Distances:", sk_distances[0])