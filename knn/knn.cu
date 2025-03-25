#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d - %s\n", \
                   __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

struct HeapNode {
    float distance;
    int index;
};

// calculate L2 distance between two points
__device__ inline float L2_distance(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

// maintain min heap
__device__ inline void heapify_up(HeapNode* heap, int pos, int* heap_size) {
    while (pos > 0) {
        int parent = (pos - 1) / 2;
        if (heap[parent].distance > heap[pos].distance) {
            HeapNode temp = heap[parent];
            heap[parent] = heap[pos];
            heap[pos] = temp;
            pos = parent;
        } else {
            break;
        }
    }
}

// kernel to make KNN
__global__ void knn_kernel(
    const float* data,
    const float* query,
    HeapNode* heap,
    int* heap_size,
    int n_points,
    int dim,
    int max_heap_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_points) return;

    float dist = L2_distance(&data[idx * dim], query, dim);
    
    int pos = atomicAdd(heap_size, 1);
    if (pos < max_heap_size) {
        heap[pos].distance = dist;
        heap[pos].index = idx;
        heapify_up(heap, pos, heap_size);
        
        if (pos == max_heap_size - 1) {
            atomicMin(heap_size, max_heap_size);
        }
    }
}

extern "C" void cuda_knn(
    const float* data, const float* query, int* indices,
    float* distances, int n_points, int dim, int k, 
    int threads
) {
    // heap size should be 2^(ceil(log2(k))+1)
    int heap_size_max = 1 << ((int)ceilf(log2f((float)k)) + 1);
    
    float *d_data, *d_query;
    HeapNode *d_heap;
    int *d_heap_size;
    
    CUDA_CHECK(cudaMalloc(&d_data, n_points * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_query, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_heap, heap_size_max * sizeof(HeapNode)));
    CUDA_CHECK(cudaMalloc(&d_heap_size, sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_data, data, n_points * dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_query, query, dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_heap_size, 0, sizeof(int)));
    
    int blocks = (n_points + threads - 1) / threads;
    knn_kernel<<<blocks, threads>>>(d_data, d_query, d_heap, d_heap_size, n_points, dim, heap_size_max);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    HeapNode* h_heap = (HeapNode*)malloc(heap_size_max * sizeof(HeapNode));
    int h_heap_size;
    CUDA_CHECK(cudaMemcpy(h_heap, d_heap, heap_size_max * sizeof(HeapNode), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_heap_size, d_heap_size, sizeof(int), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < h_heap_size - 1; i++) {
        for (int j = 0; j < h_heap_size - i - 1; j++) {
            if (h_heap[j].distance > h_heap[j + 1].distance) {
                HeapNode temp = h_heap[j];
                h_heap[j] = h_heap[j + 1];
                h_heap[j + 1] = temp;
            }
        }
    }
    
    int result_size = min(k, h_heap_size);
    for (int i = 0; i < result_size; i++) {
        distances[i] = h_heap[i].distance;
        indices[i] = h_heap[i].index;
    }
    
    free(h_heap);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_heap));
    CUDA_CHECK(cudaFree(d_heap_size));
}