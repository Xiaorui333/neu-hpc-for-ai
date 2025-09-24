// Computes in place: C <- alpha * op_s(A) * op_t(B) + beta * C
// Row-major

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

enum Transpose : int { N = 0, T = 1 };

__device__ __forceinline__
float get_elem(const float* M, int rows, int cols, int r, int c, bool trans) {
  // M is stored row-major as (rows x cols)
  return trans ? M[c * cols + r] : M[r * cols + c];
}

__global__ void gemm_inplace_kernel(const float* __restrict__ A, int a_rows, int a_cols, bool transA,
                                    const float* __restrict__ B, int b_rows, int b_cols, bool transB,
                                    float* __restrict__ C, int m, int n, int k,
                                    float alpha, float beta) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;

  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    float a = get_elem(A, a_rows, a_cols, row, kk, transA); // op(A)[row, kk]
    float b = get_elem(B, b_rows, b_cols, kk, col, transB); // op(B)[kk, col]
    acc = fmaf(a, b, acc); // acc += a*b (fused)
  }
  // in-place update of C
  C[row * n + col] = alpha * acc + beta * C[row * n + col];
}

void gemm_inplace(const float* dA, int a_rows, int a_cols, Transpose TA,
                  const float* dB, int b_rows, int b_cols, Transpose TB,
                  float* dC, float alpha, float beta) {
  // Shapes after op: op(A)=(m x k), op(B)=(k x n), C=(m x n)
  int m  = (TA == N ? a_rows : a_cols);
  int kA = (TA == N ? a_cols : a_rows);

  int kB = (TB == N ? b_rows : b_cols);
  int n  = (TB == N ? b_cols : b_rows);

  if (kA != kB) {
    printf("Dimension mismatch: kA=%d, kB=%d\n", kA, kB);
    std::exit(1);
  }
  int k = kA;

  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x,
            (m + block.y - 1) / block.y);

  gemm_inplace_kernel<<<grid, block>>>(
      dA, a_rows, a_cols, (TA == T),
      dB, b_rows, b_cols, (TB == T),
      dC, m, n, k, alpha, beta);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

int main(int argc, char** argv) {
  // Default: demonstrate A^T * B with m=4, k=3, n=5
  int a_rows = 3, a_cols = 4;  // store A as (k x m) since TA=T
  int b_rows = 3, b_cols = 5;  // store B as (k x n)
  Transpose TA = T;
  Transpose TB = N;

  if (argc >= 7) {
    a_rows = std::atoi(argv[1]);
    a_cols = std::atoi(argv[2]);
    b_rows = std::atoi(argv[3]);
    b_cols = std::atoi(argv[4]);
    /* int ta_arg = std::atoi(argv[5]); */ // TA is forced to T in this demo
    int tb_arg = std::atoi(argv[6]);
    TA = T;
    TB = (tb_arg != 0 ? T : N);
  }

  // op(A)=(m x k), op(B)=(k x n)
  int m  = (TA == N ? a_rows : a_cols);
  int kA = (TA == N ? a_cols : a_rows);
  int kB = (TB == N ? b_rows : b_cols);
  int n  = (TB == N ? b_cols : b_rows);
  if (kA != kB) { printf("Bad dims; kA=%d kB=%d\n", kA, kB); return 1; }
  int k = kA;

  const float alpha = 1.1f, beta = 0.9f;

  // Host buffers (C is m x n)
  std::vector<float> hA((size_t)a_rows * a_cols),
                     hB((size_t)b_rows * b_cols),
                     hC((size_t)m * n);

  // Deterministic init
  for (size_t i = 0; i < hA.size(); ++i) hA[i] = 0.1f * ((int)i % 7 - 3);
  for (size_t i = 0; i < hB.size(); ++i) hB[i] = 0.2f * ((int)i % 5 - 2);
  for (size_t i = 0; i < hC.size(); ++i) hC[i] = 0.05f * ((int)i % 11 - 5);

  // Device
  float *dA=nullptr, *dB=nullptr, *dC=nullptr;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size()*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size()*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dC, hC.data(), hC.size()*sizeof(float), cudaMemcpyHostToDevice));

  // Run (in-place update of C)
  gemm_inplace(dA, a_rows, a_cols, TA,
               dB, b_rows, b_cols, TB,
               dC, alpha, beta);

  // Copy back a small view
  std::vector<float> hOut(hC.size());
  CUDA_CHECK(cudaMemcpy(hOut.data(), dC, hOut.size()*sizeof(float), cudaMemcpyDeviceToHost));

  printf("GEMM in-place done. op(A)=(%d x %d), op(B)=(%d x %d), C=(%d x %d)\n",
         m, k, k, n, m, n);
  printf("Top-left 4x4 of C:\n");
  int pr = (m < 4 ? m : 4), pc = (n < 4 ? n : 4);
  for (int r = 0; r < pr; ++r) {
    for (int c = 0; c < pc; ++c) printf("%8.4f ", hOut[r*n + c]);
    printf("\n");
  }

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return 0;
}
