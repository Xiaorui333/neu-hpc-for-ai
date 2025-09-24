# Week 03 – Assignment  


## 📘 GEMM – Inplace Transpose Kernel  

Implements a matrix multiplication kernel where it can optionally transpose either $A$ or $B$, and updates $C$ in place instead of requiring an additional allocation $D$.  

- **Source Code**: [`gemm_inplace_transpose.cu`](./gemm_inplace_transpose.cu)  
- **Run**:  
  ```bash
  ./scripts/run_gpu.sh week_03/gemm_inplace_transpose.cu
  ```  



## 📘 GEMM – Tiled Kernel  

Implements a **tiled matrix multiplication kernel** that leverages shared memory to reduce HBM access.  

- **Source Code**: [`gemm_tiled.cu`](./gemm_tiled.cu)  
- **Run**:  
  ```bash
  ./scripts/run_gpu.sh week_03/gemm_tiled.cu
  ```  

---

## 📘 Online Normalizer Softmax (C)  

Implements the **online normalizer Softmax** based on *Algorithm 3* in the reference paper, ensuring numerical stability.  

- **Source Code**: [`online_softmax.c`](./online_softmax.c)  
- **Run**:  
  ```bash
  cc -O2 -DTEST_ONLINE_SOFTMAX online_softmax.c -lm && ./a.out
  ```  

