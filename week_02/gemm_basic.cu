// Computes: D = alpha * (A * B) + beta * C
// Row-major: A[m×k], B[k×n], C[m×n], D[m×n]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(x) do { \
  cudaError_t e = (x); \
  if (e != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); \
  } \
} while (0)

__global__ void gemm_basic_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  const float* __restrict__ C,
                                  float* __restrict__ D,
                                  int M, int N, int K,
                                  float alpha, float beta) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  float acc = 0.0f;
  for (int kk = 0; kk < K; ++kk) {
    acc += A[row * K + kk] * B[kk * N + col];
  }
  D[row * N + col] = alpha * acc + beta * C[row * N + col];
}

void gemm_basic(const float* dA, const float* dB, const float* dC, float* dD,
                int M, int N, int K, float alpha, float beta) {
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x,
            (M + block.y - 1) / block.y);

  gemm_basic_kernel<<<grid, block>>>(dA, dB, dC, dD, M, N, K, alpha, beta);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

int main(int argc, char** argv) {
  int M = 8, N = 8, K = 8;
  if (argc == 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }

  const float alpha = 1.1f, beta = 0.9f;

  // Host buffers
  std::vector<float> hA(M*K), hB(K*N), hC(M*N), hD(M*N);

  // Simple deterministic init 
  for (int i = 0; i < M*K; ++i) hA[i] = 0.1f * ((i % 7) - 3);
  for (int i = 0; i < K*N; ++i) hB[i] = 0.2f * ((i % 5) - 2);
  for (int i = 0; i < M*N; ++i) hC[i] = 0.05f * ((i % 11) - 5);

  // Device buffers
  float *dA=nullptr, *dB=nullptr, *dC=nullptr, *dD=nullptr;
  CUDA_CHECK(cudaMalloc(&dA, sizeof(float)*M*K));
  CUDA_CHECK(cudaMalloc(&dB, sizeof(float)*K*N));
  CUDA_CHECK(cudaMalloc(&dC, sizeof(float)*M*N));
  CUDA_CHECK(cudaMalloc(&dD, sizeof(float)*M*N));

  CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeof(float)*K*N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dC, hC.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));

  // Run
  gemm_basic(dA, dB, dC, dD, M, N, K, alpha, beta);

  // Copy back and print a small view
  CUDA_CHECK(cudaMemcpy(hD.data(), dD, sizeof(float)*M*N, cudaMemcpyDeviceToHost));

  printf("GEMM basic done. M=%d N=%d K=%d (alpha=%.2f, beta=%.2f)\n", M, N, K, alpha, beta);
  printf("Top-left 4x4 block of D:\n");
  int pr = (M < 4 ? M : 4), pc = (N < 4 ? N : 4);
  for (int r = 0; r < pr; ++r) {
    for (int c = 0; c < pc; ++c) {
      printf("%8.4f ", hD[r*N + c]);
    }
    printf("\n");
  }

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaFree(dD));
  return 0;
}
