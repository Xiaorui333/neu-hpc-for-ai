# Week 05 – Assignment  


## 📘 Implement FlashAttention-2 in CUDA  

1. Forward pass and backward pass

- **Source Code**: [`fa2_forward.cu`](./fa2_forward.cu)  
- **Run**:  
  ```bash
  ./scripts/run_gpu.sh week_05/fa2_forward.cu
  ```  

- **Source Code**: [`fa2_backward.cu`](./fa2_backward.cu)  
- **Run**:  
  ```bash
  ./scripts/run_gpu.sh week_05/fa2_backward.cu
  ```  


2. Parallelism

- **Source Code**: [`fa2_fwd_bwd_parallelism.cu`](./fa2_fwd_bwd_parallelism.cu)  
- **Run**:  
  ```bash
  ./scripts/run_gpu.sh week_05/fa2_fwd_bwd_parallelism.cu
  ```  


3. Work Partitioning between Warps

- **Source Code**: [`fa2_warp_partition.cu`](./fa2_warp_partition.cu)  
- **Run**:  
  ```bash
  ./scripts/run_gpu.sh week_05/fa2_warp_partition.cu
  ```  