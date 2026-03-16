# HPC for AI — Portfolio

![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia)
![C](https://img.shields.io/badge/C-00599C?logo=c)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C?logo=pytorch)
![MPI](https://img.shields.io/badge/MPI-HPC--X-blue)
![Modal](https://img.shields.io/badge/Cloud-Modal-black)


Modern large language models like GPT-4, LLaMA, and DeepSeek-V3 owe their speed not just to bigger GPUs, but to a handful of carefully engineered kernels — FlashAttention, tiled GEMM, Mixture-of-Experts — running across dozens of devices connected by high-speed networks.

This repository is the record of building those kernels **from scratch**, one layer at a time.

It starts with the simplest possible question: *how do you multiply two matrices in parallel?* A single-threaded C loop becomes a multi-threaded one with pthreads, which reveals the limits of CPU parallelism and motivates the move to GPU. On the GPU, a naive one-thread-per-element CUDA kernel works but wastes memory bandwidth, so shared-memory tiling is introduced — the same technique at the heart of every production GEMM. With tiling understood, the next question becomes: *what if the matrix is an attention score matrix that doesn't fit in memory?* That leads to FlashAttention, which tiles the attention computation itself and uses an online softmax to avoid ever materializing the full N-by-N matrix. FlashAttention-2 pushes further — reordering loops, partitioning work across warps, and computing gradients — to reach near-peak GPU utilization.

But a single GPU has limits. Scaling attention to longer sequences and larger models requires distributing K/V across multiple GPUs and merging results with MPI. And attention is only half the story: the other performance-critical layer in frontier models is the Mixture-of-Experts feed-forward block, where each token is routed to a small subset of expert networks. Building the DeepSeekV3-style MoE operator — router, shared expert, routed experts, all with custom CUDA kernels — reveals a new bottleneck: the all-to-all communication needed to shuffle tokens between GPUs. The final project attacks this directly with FlashMoE, fusing expert computation with one-sided RDMA via NVSHMEM so that communication and compute overlap at the hardware level.

The result is a complete vertical stack: from `for (int i = 0; ...)` in C to multi-GPU RDMA kernel fusion — every layer written, tested, and run on NVIDIA A100s.

```
CPU threads ── CUDA basics ── shared memory tiling ── online softmax
                                                           │
                        FlashAttention ── FlashAttention-2 ─┘
                                                │
                        Distributed FA-2 (MPI, multi-GPU)
                                                │
                        DeepSeekV3 MoE (custom CUDA + PyTorch)
                                                │
                        FlashMoE RDMA (NVSHMEM, fused compute + communication)
```

---

## Repository Structure

```
├── scripts/          # Build & run infrastructure (Modal, NVCC, MPI)
├── util/             # Shared C/CUDA utility headers
├── week_01/          # Modal basics, llama2.c, threaded matrix multiply
├── week_02/          # CUDA GEMM variants, matrix–vector, GPU theory
├── week_03/          # Tiled GEMM, in-place transpose, online softmax
├── week_04/          # FlashAttention (CPU + CUDA)
├── week_05/          # FlashAttention-2 (forward, backward, warp partitioning)
├── week_07/          # Distributed FlashAttention-2 with MPI
├── week_08/          # DeepSeekV3 Mixture-of-Experts with custom CUDA kernels
└── week_11/          # FlashMoE with NVSHMEM RDMA (multi-GPU)
```

---

## Week by Week

### Week 01 — Where It Begins: CPU Parallelism and the Case for GPUs

The journey starts on the CPU. A single-threaded C matrix multiply is correct but slow; adding pthreads gives near-linear speedup up to 32 threads — and then the gains plateau. This ceiling is the whole motivation for GPU programming: a CPU has tens of cores, a GPU has thousands of CUDA cores. Alongside the matrix work, the Modal cloud-GPU platform is set up (writing and deploying a hello-world function remotely), and Meta's llama2.c is run locally to see what an LLM inference workload actually looks like at the systems level.

| | |
|---|---|
| **Skillsets** | C, POSIX pthreads, gcc, Modal (serverless GPU), LLM inference (llama2.c) |
| **Impact** | Established the full development loop — write locally, run on cloud A100s — used for every project that follows. The pthreads speedup plateau makes a concrete case for moving to GPU. |

**Key files:** [`hello_world.py`](week_01/hello_world.py) · [`single_thread.c`](week_01/single_thread.c) · [`multi_thread.c`](week_01/multi_thread.c)

---

### Week 02 — First CUDA Kernels and GPU Architecture

With the CPU ceiling established, the same matrix multiply moves to the GPU. A naive CUDA GEMM kernel (D = αAB + βC) launches one thread per output element — simple, but it exposes every important GPU concept at once: thread/block/grid organization, warp divergence, SIMD efficiency, memory coalescing, occupancy, and register pressure. Comparing a row-per-thread kernel against a column-per-thread kernel reveals that **memory access pattern matters more than raw parallelism**: column-per-thread coalesces writes and reads of B, while row-per-thread does not. These exercises — warp divergence analysis, occupancy calculations under register and shared-memory limits — form the mental model for all kernel optimization to come.

| | |
|---|---|
| **Skillsets** | CUDA C++, NVCC, GPU thread/block/grid model, warp-level analysis, memory coalescing, occupancy tuning, Java |
| **Impact** | Built the foundational understanding of GPU execution. The coalescing and occupancy analyses directly informed every kernel design decision in subsequent weeks. |

**Key files:** [`gemm_basic.cu`](week_02/gemm_basic.cu) · [`0301_mm_row_per_thread.cu`](week_02/0301_mm_row_per_thread.cu) · [`0301_mm_col_per_thread.cu`](week_02/0301_mm_col_per_thread.cu) · [`0302_matvec.cu`](week_02/0302_matvec.cu) · [`Llama2Loader.java`](week_02/Llama2Loader.java)

---

### Week 03 — Tiling, Shared Memory, and Online Softmax

The naive GEMM from Week 02 reads every element of A and B from global memory N times — a massive waste of bandwidth. The fix is **tiling**: load a 32×32 block of A and B into fast on-chip shared memory, compute the partial product, then move to the next tile. This single idea reduces global memory traffic by 32× and is the core of every high-performance GPU kernel. An in-place transpose variant adds flexibility for different matrix layouts.

But GEMM alone isn't enough. The next target is attention, and attention requires softmax. The standard two-pass softmax (find max, then normalize) requires reading the entire row twice. The **online normalizer softmax** (Algorithm 3 from the FlashAttention paper) computes the result in a single pass by maintaining a running max and normalizer — a key building block for what comes next.

| | |
|---|---|
| **Skillsets** | CUDA shared memory, `__syncthreads()`, tiling strategies, numerical stability, online algorithms |
| **Impact** | Tiled GEMM demonstrates the optimization principle behind every production GPU kernel. Online softmax is the prerequisite that makes FlashAttention possible — without it, attention cannot be computed tile-by-tile. |

**Key files:** [`gemm_tiled.cu`](week_03/gemm_tiled.cu) · [`gemm_inplace_transpose.cu`](week_03/gemm_inplace_transpose.cu) · [`online_softmax.c`](week_03/online_softmax.c)

---

### Week 04 — FlashAttention: Tiling Meets Attention

With tiled GEMM and online softmax in hand, the two ideas combine into **FlashAttention**. Standard attention computes Q·K^T (an N×N matrix), applies softmax, then multiplies by V — requiring O(N²) memory that blows up for long sequences. FlashAttention never materializes the full attention matrix. Instead, it tiles Q, K, and V, computes partial attention scores in shared memory, and uses the online softmax to accumulate the correct result tile by tile.

The implementation starts as a CPU reference in C (to get the algorithm right without GPU complexity) and then moves to a full CUDA kernel with shared memory staging, producing both the output O and the logsumexp L needed for the backward pass.

| | |
|---|---|
| **Skillsets** | Attention mechanism internals, IO-aware algorithm design, CUDA shared memory tiling, FlashAttention paper implementation |
| **Impact** | FlashAttention reduces attention memory from O(N²) to O(N), enabling longer sequence lengths in transformers. This is the single most impactful kernel optimization in modern LLM training — the same algorithm inside PyTorch's `scaled_dot_product_attention`. |

**Key files:** [`flash_attention_tile.c`](week_04/flash_attention_tile.c) · [`flash_attention_cuda.cu`](week_04/flash_attention_cuda.cu)

---

### Week 05 — FlashAttention-2: Faster Forward, Full Backward, Warp-Level Optimization

FlashAttention works, but leaves performance on the table. FlashAttention-2 fixes this with three changes: (1) swap the loop order so the outer loop iterates over Q blocks and the inner loop over K/V — this keeps Q in registers and streams K/V through shared memory, reducing non-matmul FLOPs; (2) partition work across warps within a thread block using `__shfl_down_sync` for efficient intra-warp reductions; (3) implement the full backward pass to compute dQ, dK, dV gradients, making the kernel usable for training, not just inference.

The result is four separate kernels that progressively add complexity: forward, backward, forward-backward parallelism across sequence blocks, and warp-level partitioning.

| | |
|---|---|
| **Skillsets** | FlashAttention-2 algorithm, CUDA warp primitives (`__shfl_down_sync`), gradient computation, forward/backward parallelism |
| **Impact** | FA-2 achieves ~2× speedup over FA-1. The warp partitioning is the same technique used in the official FA-2 kernel, maximizing occupancy and minimizing synchronization — production-grade optimization. |

**Key files:** [`fa2_forward.cu`](week_05/fa2_forward.cu) · [`fa2_backward.cu`](week_05/fa2_backward.cu) · [`fa2_fwd_bwd_parallelism.cu`](week_05/fa2_fwd_bwd_parallelism.cu) · [`fa2_warp_partition.cu`](week_05/fa2_warp_partition.cu)

---

### Week 07 — Going Multi-GPU: Distributed FlashAttention-2 with MPI

A single A100 can only hold so much of K and V. For longer sequences and larger models, the attention computation must be split across GPUs. This week extends FA-2 to a distributed setting: K and V are sharded across MPI ranks, each rank runs local FA-2 on its shard, and then an MPI reduction merges the per-rank running max (m) and normalizer (l) into a globally correct result.

The tricky part is the merge: combining partial online softmax statistics from different ranks requires carefully rescaling partial outputs — a numerically delicate operation. The implementation runs on 1–8 A100 GPUs via Modal with HPC-X (CUDA-aware MPI over UCX).

| | |
|---|---|
| **Skillsets** | MPI (HPC-X), CUDA-aware MPI, UCX transport, distributed attention, multi-GPU synchronization |
| **Impact** | This is the sequence-parallel attention strategy used in Megatron-LM and DeepSpeed. Running across 8 A100s validates the correctness of distributed online softmax merging — a non-trivial numerical challenge at the core of large-scale LLM training. |

**Key files:** [`dist_fa2_forward_mpi.cu`](week_07/dist_fa2_forward_mpi.cu)

```bash
python3.11 -m modal run scripts/modal_nvcc_mpi.py::compile_and_run_cuda \
  --code-path week_07/dist_fa2_forward_mpi.cu --np 6
```

---

### Week 08 — DeepSeekV3 Mixture-of-Experts: A New Bottleneck

Attention is now fast and distributed, but in frontier models like DeepSeek-V3, Mixtral, and Switch Transformer, the other performance-critical layer is the **Mixture-of-Experts (MoE)** feed-forward block. Instead of one large FFN applied to every token, MoE routes each token to its top-K experts out of many (e.g., 2 out of 64), achieving GPT-4-level quality at a fraction of the compute.

This week builds the full DeepSeekV3-style MoE operator from scratch: a Top-K router that computes gating logits and selects experts, a shared expert applied to all tokens, and routed experts with Swish activation. Every component has both a PyTorch fallback and a custom CUDA kernel (router forward, top-k selection, shared expert, routed expert), exposed to Python via pybind11 and built with setuptools.

The implementation works — but it reveals a new problem. In a distributed setting, tokens must be shuffled between GPUs to reach their assigned experts (all-to-all communication), and this shuffle becomes the dominant cost.

| | |
|---|---|
| **Skillsets** | PyTorch custom CUDA extensions, pybind11, MoE architecture (Top-K routing, Swish gating), setuptools C++ build |
| **Impact** | MoE is the architecture enabling frontier models to scale quality without scaling compute linearly. This implementation covers the full route-dispatch-compute-combine pipeline, revealing the all-to-all communication bottleneck that motivates the final project. |

**Key files:** [`deepseek_moe.py`](week_08/deepseek_moe.py) · [`deepseek_moe_cuda.cu`](week_08/deepseek_moe_cuda.cu) · [`deepseek_moe.cpp`](week_08/deepseek_moe.cpp) · [`setup.py`](week_08/setup.py) · [`modal_deepseek_simple.py`](week_08/modal_deepseek_simple.py)

---

### Week 11 — FlashMoE: Fusing Compute and Communication with RDMA

The all-to-all bottleneck exposed in Week 08 is the final boss. Standard distributed MoE separates communication (shuffle tokens to experts) from computation (run experts) into two sequential phases — the GPU sits idle while waiting for data, and the network sits idle during computation.

FlashMoE eliminates this by **fusing** expert computation with one-sided RDMA communication via NVSHMEM. While one expert is computing, the next batch of tokens is already being fetched from a remote GPU's memory through direct GPU-to-GPU RDMA — no CPU involvement, no synchronization barriers. This is the approach described in the FlashMoE paper (ICML 2025).

The implementation includes NVSHMEM source compilation and installation on Modal, PyTorch CUDA extension bindings for the fused kernel, an MPI-based multi-PE launcher with strict PE-GPU binding, and both single-GPU and multi-GPU test harnesses running on A100s.

| | |
|---|---|
| **Skillsets** | NVSHMEM, RDMA, multi-GPU communication, MPI + NVSHMEM co-design, PyTorch C++ extensions, Modal multi-GPU orchestration |
| **Impact** | FlashMoE with RDMA represents the cutting edge of distributed MoE — overlapping compute and communication at the hardware level. This is where kernel engineering meets network engineering, and it's the technique that makes efficient large-scale MoE training possible. |

**Key files:** [`flashmoe_bindings_rdma.cpp`](week_11/flashmoe_bindings_rdma.cpp) · [`setup_flashmoe_rdma.py`](week_11/setup_flashmoe_rdma.py) · [`modal_flashmoe.py`](week_11/modal_flashmoe.py)

---

## Infrastructure

### Build & Run Scripts

| Script | Purpose |
|--------|---------|
| [`scripts/modal_nvcc.py`](scripts/modal_nvcc.py) | Compile and run single-GPU CUDA on Modal (A100-40GB, CUDA 12.8) |
| [`scripts/modal_nvcc_mpi.py`](scripts/modal_nvcc_mpi.py) | Compile and run MPI+CUDA with HPC-X and CUDA-aware MPI |
| [`scripts/run_gpu.sh`](scripts/run_gpu.sh) | Shortcut: `modal run scripts/modal_nvcc.py --code-path <file.cu>` |
| [`scripts/run_cpu.sh`](scripts/run_cpu.sh) | Run C code with clang + AddressSanitizer |

### Shared Utility Headers (`util/`)

Reusable C/CUDA headers used across projects: tile abstraction (`tile.h`), matrix utilities (`mat_util.h`), ceiling division (`div_ceil.h`), float comparison (`float_eq.h`), CUDA compatibility shims (`cuda_shim.h`), timing, tracing, and more.

---

## Technology Summary

| Category | Technologies |
|----------|-------------|
| **Languages** | C, C++, CUDA, Python, Java |
| **GPU Programming** | CUDA 12.x, shared memory tiling, warp primitives, kernel fusion |
| **AI/ML Kernels** | FlashAttention, FlashAttention-2, online softmax, Mixture-of-Experts |
| **Distributed Systems** | MPI (HPC-X), NVSHMEM, RDMA, UCX, multi-GPU synchronization |
| **Frameworks** | PyTorch, pybind11, setuptools C++ extensions |
| **Cloud Infrastructure** | Modal (serverless GPU), NVIDIA A100 |
| **Build Tools** | NVCC, gcc/g++, CMake, ninja |
