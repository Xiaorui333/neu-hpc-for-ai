# Week 02 – Assignment


## 📘 Chapter 3 — Exercises

1. In this chapter we implemented a matrix multiplication kernel that has each thread produce one output matrix element. In this question, you will implement different matrix-matrix multiplication kernels and compare them.  
   a. **One thread per output row**  
      **Source Code**: [0301_mm_row_per_thread.cu](./0301_mm_row_per_thread.cu)  

   b. **One thread per output column**   
      **Source Code**: [0301_mm_col_per_thread.cu](./0301_mm_col_per_thread.cu)  

   c. **Pros and cons (row‑major arrays)**  
      • Thread = **row**: + reuses A’s row (register/L1 friendly); − cross‑thread reads stride **K** and writes stride **N** (poor coalescing); parallel width only **M**.  
      • Thread = **column**: + coalesced reads of **B** and coalesced writes to **C**; − A loads are broadcast; parallel width only **N**.  
      • Rule of thumb: column‑per‑thread often better in row‑major; for performance use **tiled GEMM with shared memory**.

2. A matrix‑vector multiplication … Write a kernel and host stub… (square, float; one thread computes one output).  
   **Source Code**: [0302_matvec.cu](./0302_matvec.cu) 

3. Consider the following CUDA kernel and the corresponding host function that calls it (from the provided image).  
   a. **Threads per block:** 16 * 32 * 1 = 512  
   b. **Threads in the grid:** gridDim.x = ceil(N/16) = ceil(300/16) = 19; 
  gridDim.y = ceil(M/32) =  ceil(150/32) = 5
  ⇒ 19 * 5 * 512 = 48,640
   c. **Blocks in the grid:** 19 * 5 = 95  
   d. **Threads that execute line 05:** 150 * 300 = 45,000 (only those with `row < M && col < N`)

4. Consider a 2D matrix with width 400 and height 500… element at row 20 and column 10.  
   a. **Row‑major:** `index = row*width + col = 20*400 + 10 = 8010`  
   b. **Column‑major:** `index = col*height + row = 10*500 + 20 = 5020`

5. Consider a 3D tensor with width 400, height 500, depth 300… element at (x=10, y=20, z=5), row‑major.  
   • **Index:** `z*(H*W) + y*W + x = 5*(500*400) + 20*400 + 10 = 1,008,010`



## 📘 Chapter 4 — Exercises

1. Consider the following CUDA kernel and the corresponding host function that calls it (image).  
   a. **Warps per block:** `ceil( blockDim.x * blockDim.y * blockDim.z / 32 )`  
   b. **Warps in the grid:** `warps_per_block * (gridDim.x * gridDim.y * gridDim.z)`  
   c. For the statement on line 04:  
      i. **Active warps:** number of warps with ≥ 1 active lane at that line.  
      ii. **Divergent warps:** warps with 0 < active lanes < 32.  
      iii. **SIMD efficiency of warp 0, block 0:** `(active_lanes / 32) × 100%`.  
      iv. **SIMD efficiency of warp 1, block 0:** same formula.  
      v. **SIMD efficiency of warp 3, block 0:** same formula.  
   d. For the statement on line 07:  
      i. **Active warps:** warps with ≥ 1 active lane at that line.  
      ii. **Divergent warps:** warps with 0 < active lanes < 32.  
      iii. **SIMD efficiency of warp 0, block 0:** `(active_lanes / 32) × 100%`.  
   e. For the loop on line 09:  
      i. **Iterations with no divergence:** iterations where every active warp has 0 or 32 active lanes.  
      ii. **Iterations with divergence:** iterations where at least one active warp has a mixed lane mask.

2. For a vector addition (N = 2000), each thread computes one output, block size = 512.  
   • **Threads in the grid:** `ceil(2000/512) * 512 = 4 * 512 = 2,048`.

3. For the previous question, boundary‑check divergence.  
   • **Divergent warps:** **1** (in the last block; cutoff inside one warp).

4. Hypothetical block with 8 threads; section times (μs): 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9.  
   • \(T_{max}=3.0\), \(\sum t_i=19.9\); wait \(=8×3.0−19.9=4.1\).  
   • **Waiting percentage:** \(4.1/24.0 ≈ 17.1\%\).

5. Programmer claims: “If I launch only 32 threads per block, I can omit `__syncthreads()` anywhere.”  
   • **Answer:** Not advisable. Correctness for block‑wide shared memory still requires `__syncthreads()`; newer GPUs use independent thread scheduling. Prefer `__syncthreads()` (block) or `__syncwarp()` (warp) appropriately.

6. Device limits: up to 1,536 threads/SM and up to 4 blocks/SM. Which config yields most threads?  
   a. 128 tpb → 4 blocks ⇒ 512 threads  
   b. 256 tpb → 4 blocks ⇒ 1,024 threads  
   c. 512 tpb → min(4, floor(1536/512)=3) ⇒ **1,536 threads (max)** ✅  
   d. 1024 tpb → only 1 block ⇒ 1,024 threads  
   • **Best:** (c) 512 threads per block.

7. Device caps: up to 64 blocks/SM and 2,048 threads/SM. Possible assignments and occupancy?  
   a. 8 blocks × 128 thr = **1,024** ⇒ **possible**, **50%** occupancy  
   b. 16 × 64 = **1,024** ⇒ **possible**, **50%**  
   c. 32 × 32 = **1,024** ⇒ **possible**, **50%**  
   d. 64 × 32 = **2,048** ⇒ **possible**, **100%**  
   e. 32 × 64 = **2,048** ⇒ **possible**, **100%**

8. Hardware limits: 2,048 threads/SM, 32 blocks/SM, 65,536 regs/SM. Full occupancy? If not, limiting factor.  
   a. 128 tpb, 30 regs/t: need 16 blocks for 2,048 threads; regs = 16×128×30 = **61,440** ≤ 65,536 ⇒ **Full occupancy**.  
   b. 32 tpb, 29 regs/t: need 64 blocks but max **32** ⇒ ≤1,024 threads ⇒ **Not full; block‑count limited**.  
   c. 256 tpb, 34 regs/t: need 8 blocks; regs = 8×256×34 = **69,632** > 65,536 ⇒ **Not full; register‑limited**. Max blocks = 65,536/(256×34) = 7 ⇒ **1,792 threads (~87.5%)**.

9. Student: 1024×1024 GEMM with 32×32 thread blocks on a device capped at 512 threads/block; one thread computes one C element.  
   • **Reaction:** 32×32 = 1,024 threads/block > 512 limit → **not possible** on that device. Must use ≤512 tpb (e.g., 16×32 or 32×16) or different tiling.



## 📘 Chapter 5 — Exercises

1. Consider matrix addition. Can one use shared memory to reduce the global memory bandwidth consumption?  
   **Answer:** **No.** Each thread reads its own `A[i]` and `B[i]` exactly once to produce `C[i]`. There is no inter‑thread reuse; staging in shared memory only adds traffic and sync overhead.

2. Draw the equivalent of Fig. 5.7 for an 8×8 GEMM with 2×2 tiling and 4×4 tiling. Verify the bandwidth reduction.  
   a. **2×2 tiling:** each element loaded once and reused by **T=2** threads ⇒ **2×** reduction of global traffic for `A` and for `B`.  
   b. **4×4 tiling:** each element reused by **T=4** threads ⇒ **4×** reduction.  
   **Conclusion:** reduction ∝ tile dimension **T**.

3. What happens if one or both `__syncthreads()` are omitted in Fig. 5.9?  
   **Answer:** Threads may read uninitialized/partial tiles or overwrite tiles still in use.

4. Registers vs. shared memory (capacity not an issue): why prefer shared memory for values fetched from global memory?  
   **Answer:** Registers are thread‑private; shared memory is block‑shared, enabling **inter‑thread reuse** (tiling) and reducing global traffic.

5. For the tiled GEMM, using a **32×32** tile, what is the reduction of global bandwidth for inputs **M** and **N**?  
   **Answer:** **32×** reduction **for each** of `M` and `N`.

6. Kernel launch: **1000 blocks × 512 threads**. If a variable is **local** in the kernel, how many versions exist over the run?  
   **Answer:** **1000 × 512 = 512,000** (one per thread).

7. Same launch; if a variable is in **shared memory**, how many versions exist?  
   **Answer:** **1000** (one per block).

8. GEMM of two **N×N** matrices. How many times is each input element requested from global memory?  
   a. **No tiling:** **N** times per element (both `A` and `B`).  
   b. **T×T tiling:** **N/T** times per element (both `A` and `B`).

9. A kernel performs **36 FLOPs** and **seven 32‑bit global reads/writes** per thread. Classify boundness.  
   Arithmetic intensity **AI = 36 / (7×4) = 36/28 ≈ 1.286 FLOPs/Byte**.  
   a. Peak 200 GFLOPS, 100 GB/s → peak ratio **2.0 FLOPs/B**  → **memory‑bound** (1.286 (36/(4 * 7 )) < 2.0).  
   b. Peak 300 GFLOPS, 250 GB/s → peak ratio **1.2 FLOPs/B** → **compute‑bound** (1.286 > 1.2).

10. Tile transpose kernel (`BLOCK_WIDTH` ∈ [1,20]).  
    a. **Correctness range:** Works only when the whole block fits **one warp** → `BLOCK_WIDTH² ≤ 32` ⇒ **{1,2,3,4,5}** (accidental warp‑synchronous behavior).  
    b. **Root cause & fix:** Missing synchronization when reading/writing the same tile.  
       **Fix:** stage via shared memory **+ `__syncthreads()`**, or write to a **separate output** matrix:
       ```cuda
       __shared__ float tile[BLOCK_WIDTH][BLOCK_WIDTH];
       tile[ty][tx] = A[row*ld + col];
       __syncthreads();
       A[rowT*ld + colT] = tile[tx][ty];
       ```

11. Consider the given kernel & host (as in the book).  
    a. **Versions of `i` (local):** **#threads in grid → 1024**.  
    b. **Versions of `x[]`: one array per thread → 1024.  
    c. **Versions of `y_s` (shared scalar):** **#blocks → 8**.  
    d. **Versions of `b_s[]` (shared array):** **#blocks → 8**.  
    e. **Shared memory per block (bytes):** `sizeof(y_s) + len(b_s)*sizeof(b_s[0]) = 4 + 128*4 = 516`.  
    f. **FLOP/Byte ratio:** Global memory per thread: a (4 reads) + b (1 read + 1 write) = 6×4 B = 24 B.FLOPs per thread: 5 mul + 5 add = 10 FLOPs.OP/B = 10 / 24 ≈ 0.417 FLOPs/B.

12. GPU limits: **2048 thr/SM, 32 blocks/SM, 65,536 regs/SM, 96 KB shared/SM**. Can the kernel reach full occupancy?  
    a. **64 thr/block, 27 regs/thr, 4 KB shared/block** → need 32 blocks, but `32×4 KB = 128 KB > 96 KB` ⇒ **not full (shared‑mem limited)**. Max 24 blocks ⇒ **1,536 thr/SM (~75%)**.  
    b. **256 thr/block, 31 regs/thr, 8 KB shared/block** → registers allow 8 blocks (63,488 regs), shared allows 12; **8×256=2,048 thr** ⇒ **full occupancy**.




## 📘 Implement a GEMM kernel in CUDA 
**Source Code**: [gemm_basic.cu](./gemm_basic.cu)  




## 📘 Reimplement the read_checkpoint function in run.c in Java
**Source Code**: [Llama2Loader.java](./Llama2Loader.java) 