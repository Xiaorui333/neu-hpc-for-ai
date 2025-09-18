#include <stdio.h>
#include <cuda_runtime.h>

// A(N) = B(N×N) * C(N) ,row-major
__global__ void matvec_kernel(const float *B, const float *C, float *A, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float acc = 0.0f;
    for (int j=0; j<N; ++j) acc += B[i * N + j] * C[j];
    A[i] = acc;
}

int main() {
    int N = 8; 
    size_t sizeB = (size_t)N * N * sizeof(float);
    size_t sizeC = (size_t)N * sizeof(float);
    size_t sizeA = (size_t)N * sizeof(float);

    float *hB = (float*)malloc(sizeB);
    float *hC = (float*)malloc(sizeC);
    float *hA = (float*)malloc(sizeA);

    for (int i = 0; i < N*N; ++i) hB[i] = 1.0f;
    for (int j = 0; j < N;   ++j) hC[j] = 1.0f;

    float *dB, *dC, *dA;
    cudaMalloc(&dB, sizeB); cudaMalloc(&dC, sizeC); cudaMalloc(&dA, sizeA);
    cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, sizeC, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    matvec_kernel<<<blocks, threads>>>(dB, dC, dA, N);
    cudaDeviceSynchronize();

    cudaMemcpy(hA, dA, sizeA, cudaMemcpyDeviceToHost);
    printf("Result A (size=%d):\n", N);
    for (int i = 0; i < N; ++i) printf("%5.1f%s", hA[i], (i+1==N? "\n":" "));
   

    cudaFree(dB); cudaFree(dC); cudaFree(dA);
    free(hB); free(hC); free(hA);
    return 0;
}