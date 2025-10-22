# Week 07 – Assignment  


## 📘 Implement distributed FlashAttentionV2 for multiple GPUs 

1. Forward pass

- **Source Code**: [`dist_fa2_forward_mpi.cu`](./dist_fa2_forward_mpi.cu)  
- **Run**:  
  ```bash
  python3.11 -m modal run scripts/modal_nvcc.py \ --code-path week_07/dist_fa2_forward_mpi.cu \ --np 6
  ```  


2. HPC-X configuration for CUDA-aware MPI

- **Source Code**: [`fa2_fwd_bwd_parallelism.cu`](../scripts/modal_nvcc.py)  