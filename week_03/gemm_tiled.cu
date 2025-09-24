// Tiled GEMM: C <- alpha * op(A) * op(B) + beta * C
// Row-major

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#ifndef TILE
#define TILE 32
#endif

#define CUDA_CHECK(x) do { \
  cudaError_t e = (x); \
  if (e != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); \
  } \
} while (0)

enum Transpose : int { N = 0, T = 1 };

// Map op(M)[r,c] -> stored index in row-major M(rows x cols)
__device__ __forceinline__
int idx_op(int rows, int cols, int r, int c, bool trans) {
  return trans ? (c * cols + r) : (r * cols + c);
}

__global__ void gemm_tiled_kernel(const float* __restrict__ A, int a_rows, int a_cols, bool transA,
                                  const float* __restrict__ B, int b_rows, int b_cols, bool transB,
                                  float* __restrict__ C, int m, int n, int k,
                                  float alpha, float beta) {
  // Each block computes a TILE x TILE tile of C (in op-shape space)
  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;

  __shared__ float As[TILE][TILE + 1];
  __shared__ float Bs[TILE][TILE + 1];

  float acc = 0.0f;

  // Iterate over tiles along the inner dimension k
  const int numTiles = (k + TILE - 1) / TILE;
  for (int t = 0; t < numTiles; ++t) {
    const int kkA = t * TILE + threadIdx.x; 
    const int kkB = t * TILE + threadIdx.y; 

    // --- Load a TILE of op(A) into shared memory ---
    float a_val = 0.0f;
    if (row < m && kkA < k) {
      const int ra = transA ? kkA : row;
      const int ca = transA ? row : kkA;
      const int ai = ra * a_cols + ca;
      a_val = A[ai];
    }
    As[threadIdx.y][threadIdx.x] = a_val;

    // --- Load a TILE of op(B) into shared memory ---
    float b_val = 0.0f;
    if (kkB < k && col < n) {
      const int rb = transB ? col : kkB;  
      const int cb = transB ? kkB : col;  
      const int bi = rb * b_cols + cb;    
      b_val = B[bi];
    }
    Bs[threadIdx.y][threadIdx.x] = b_val;

    __syncthreads();

    // --- Compute partial products for this tile ---
    #pragma unroll
    for (int kk = 0; kk < TILE; ++kk) {
      acc = fmaf(As[threadIdx.y][kk], Bs[kk][threadIdx.x], acc);
    }
    __syncthreads();
  }

  // --- Write back in place ---
  if (row < m && col < n) {
    const int ci = row * n + col;
    C[ci] = alpha * acc + beta * C[ci];
  }
}

void gemm_tiled_inplace(const float* dA, int a_rows, int a_cols, Transpose TA,
                        const float* dB, int b_rows, int b_cols, Transpose TB,
                        float* dC, float alpha, float beta) {
  // Shapes after op: op(A)=(m x k), op(B)=(k x n), C=(m x n)
  const int m  = (TA == N ? a_rows : a_cols);
  const int kA = (TA == N ? a_cols : a_rows);
  const int kB = (TB == N ? b_rows : b_cols);
  const int n  = (TB == N ? b_cols : b_rows);

  if (kA != kB) {
    printf("Dimension mismatch: kA=%d, kB=%d\n", kA, kB);
    std::exit(1);
  }
  const int k = kA;

  dim3 block(TILE, TILE);
  dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);

  gemm_tiled_kernel<<<grid, block>>>(
      dA, a_rows, a_cols, (TA == T),
      dB, b_rows, b_cols, (TB == T),
      dC, m, n, k, alpha, beta);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

int main(int argc, char** argv) {
  // Example: op(A)=(128x96), op(B)=(96x64), C=(128x64)
  // Store A as 128x96 (TA=N), store B as 64x96 (TB=T)
  const int a_rows = 128, a_cols = 96;  const Transpose TA = N;
  const int b_rows =  64, b_cols = 96;  const Transpose TB = T;

  // op(A)=(m x k), op(B)=(k x n)
  const int m  = (TA == N ? a_rows : a_cols);
  const int kA = (TA == N ? a_cols : a_rows);
  const int kB = (TB == N ? b_rows : b_cols);
  const int n  = (TB == N ? b_cols : b_rows);
  if (kA != kB) { printf("Bad dims; kA=%d kB=%d\n", kA, kB); return 1; }
  const int k = kA;

  const float alpha = 1.0f, beta = 0.0f;

  // Host buffers (C is m x n)
  std::vector<float> hA((size_t)a_rows * a_cols),
                     hB((size_t)b_rows * b_cols),
                     hC((size_t)m * n);

  // Deterministic init
  for (size_t i = 0; i < hA.size(); ++i) hA[i] = 0.1f * ((int)i % 7 - 3);
  for (size_t i = 0; i < hB.size(); ++i) hB[i] = 0.2f * ((int)i % 5 - 2);
  for (size_t i = 0; i < hC.size(); ++i) hC[i] = 0.0f;

  // Device
  float *dA=nullptr, *dB=nullptr, *dC=nullptr;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size()*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size()*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dC, hC.data(), hC.size()*sizeof(float), cudaMemcpyHostToDevice));

  gemm_tiled_inplace(dA, a_rows, a_cols, TA,
                     dB, b_rows, b_cols, TB,
                     dC, alpha, beta);

  // Copy back a small view
  std::vector<float> hOut(hC.size());
  CUDA_CHECK(cudaMemcpy(hOut.data(), dC, hOut.size()*sizeof(float), cudaMemcpyDeviceToHost));

  printf("TILED GEMM in-place done. op(A)=(%d x %d), op(B)=(%d x %d), C=(%d x %d), TILE=%d\n",
         m, k, k, n, m, n, TILE);
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
