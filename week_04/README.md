# Week 04 – Assignment  


## Chapter 6 - Exercise 4. OP/B of matrix–matrix multiplication kernels

### (a) Naïve kernel (Chapter 3), no optimizations
Per output element: `2K` FLOPs (K mul + K add), global traffic ≈ `(2K reads + 1 write) × 4 B`  
\[
\text{ratio} = \frac{2K}{4(2K+1)} \xrightarrow[K\gg1]{} \frac{1}{4} \approx \mathbf{0.25\ OP/B}.
\]

### (b) Tiled kernel (Chapter 5) with 32×32 shared-memory tiling
Each input element of A/B is reused `T` times from shared memory; AI scales by `T` vs naïve.  
For `T = 32`:
\[
\text{ratio} \approx \frac{T}{4} = \frac{32}{4} = \mathbf{8\ OP/B}.
\]

### (c) Tiled (32×32) + thread coarsening factor 4 (Chapter 6)
Coarsening increases per-thread work but **does not change** global A/B/C traffic at the tile level; FLOPs and global bytes per tile-step are essentially unchanged.  
\[
\text{ratio} = \mathbf{8\ OP/B}.
\]





## 📘 Implement flash attention in CUDA  

1. Simpler attention implementation in C

- **Source Code**: [`flash_attention_tile.c`](./flash_attention_tile.c)  
- **Run**:  
  ```bash
  gcc -O2 -std=c11 -I. -o flash_attention \
  week_04/flash_attention_tile.c \
  -lm

  ./flash_attention
  ```  

2. Fast and Memory-Efficient Exact Attention with IO-Awareness in CUDA

- **Source Code**: [`flash_attention_cuda.c`](./flash_attention_cuda.cu)  
- **Run**:  
  ```bash
  ./scripts/run_gpu.sh week_04/flash_attention_cuda.cu
  ```  