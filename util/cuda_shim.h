#pragma once
#include "assertc.h"

#if CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

dim3 new_dim3(unsigned int x, unsigned int y, unsigned int z) {
    dim3 d(x, y, z);
    return d;
}



#else
#include <stdlib.h>
#include <string.h>
#define __global__
#define __device__
#define __host__
#define __shared__

enum cudaError {
    cudaSuccess = 0,
    cudaErrorMemoryAllocation,
};
/* DEVICE_BUILTIN */

typedef enum cudaError cudaError_t;


typedef struct {
    unsigned int x;
    unsigned int y;
    unsigned int z;
} dim3;

dim3 new_dim3(unsigned int x, unsigned int y, unsigned int z) {
    dim3 d = {x, y, z};
    return d;
}

typedef enum cudaError cudaError_t;

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    *devPtr = malloc(size);
    assertc(*devPtr != 0);
    return cudaSuccess;
}

cudaError_t cudaFree(void* devPtr) {
    free(devPtr);
    return cudaSuccess;
}

static const struct { cudaError_t code; const char* name; const char* desc; } cuda_error_map[] = {
    { cudaSuccess,                        "cudaSuccess",                       "no error" },
    { cudaErrorMemoryAllocation,         "cudaErrorMemoryAllocation",         "out of memory" },
    { (cudaError_t)-1,                   NULL,                                NULL }  // terminator
};

const char* cudaGetErrorName(cudaError_t err) {
    for (int i = 0; cuda_error_map[i].name; ++i)
        if (cuda_error_map[i].code == err)
            return cuda_error_map[i].name;
    return "unrecognized error code";
}

const char* cudaGetErrorString(cudaError_t err) {
    for (int i = 0; cuda_error_map[i].desc; ++i)
        if (cuda_error_map[i].code == err)
            return cuda_error_map[i].desc;
    return "unrecognized error code";
}

dim3 threadIdx = {0, 0, 0};
dim3 blockIdx = {0, 0, 0};
dim3 blockDim = {0, 0, 0};
dim3 gridDim = {0, 0, 0};


typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
} cudaMemcpyKind;

cudaError_t cudaMemcpy(void *dst, const void *src, unsigned int size, cudaMemcpyKind kind) {
    memcpy(dst, src, size);
    return cudaSuccess;
}

cudaError_t cudaGetLastError(void) {
    return cudaSuccess;
}

typedef struct CUstream_st* cudaStream_t;

cudaError_t  cudaLaunchKernel(
    const void* func,        // pointer to __global__ function
    dim3 gridDim,            // grid dimensions
    dim3 blockDim,           // block dimensions
    void** args,             // array of pointers to arguments
    size_t sharedMem = 0,    // dynamic shared memory size in bytes
    cudaStream_t stream = 0) {
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void) {
    return cudaSuccess;
}

void __syncthreads(void) {

}

cudaError_t cudaGetDeviceCount(int *count) {
    return cudaSuccess;
}


cudaError_t cudaSetDevice(int device) {
    return cudaSuccess;
}

cudaError_t cudaMemGetInfo(size_t* free, size_t* total) {
    return cudaSuccess;
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count);

typedef struct cudaDeviceProp {

} cudaDeviceProp;

cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device);

#endif

#define chk(call) \
do { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
cudaGetErrorString(err)); \
assertc(0); \
} \
} while(0)

