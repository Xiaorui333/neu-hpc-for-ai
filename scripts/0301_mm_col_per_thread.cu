#include <stdio.h>
#include <cuda_runtime.h>

__global__ void mm_col_per_thread(const float* A, const float* B, float* C,
                                  int M, int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    for (int i = 0; i < M; ++i) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[i*K + k] * B[k*N + col];
        }
        C[i*N + col] = acc;
    }
}

int main() {
    // example：A(4x3) * B(3x5) = C(4x5)
    int M = 4, K = 3, N = 5;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float hA[M*K], hB[K*N], hC[M*N];

    for (int i = 0; i < M*K; ++i) hA[i] = 1.0f;
    for (int i = 0; i < K*N; ++i) hB[i] = 1.0f;

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeA);
    cudaMalloc(&dB, sizeB);
    cudaMalloc(&dC, sizeC);

    cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);

    
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    mm_col_per_thread<<<blocks, threads>>>(dA, dB, dC, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost);

    printf("Result matrix C (M=%d, N=%d):\n", M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%5.1f ", hC[i*N + j]);
        }
        printf("\n");
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
