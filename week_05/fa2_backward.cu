#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <math_constants.h>

#define CHECK_CUDA(x) do{auto e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

#ifndef USE_SCALE
#define USE_SCALE 1
#endif

// ---------- Kernel 0: D = rowsum(dO ⊙ O) ----------
__global__ void rowsum_dot_kernel(const float* __restrict__ dO,
                                  const float* __restrict__ O,
                                  float* __restrict__ D,
                                  int N, int d)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    const float* do_ptr = dO + row * d;
    const float*  o_ptr =  O + row * d;
    float acc = 0.f;
    for (int t=0; t<d; ++t) acc += do_ptr[t] * o_ptr[t];
    D[row] = acc;
}

// ---------- Kernel 1: FA-2 Backward ----------
__global__ void fa2_backward_kernel(
    const float* __restrict__ Q,    // [N,d]
    const float* __restrict__ K,    // [N,d]
    const float* __restrict__ V,    // [N,d]
    const float* __restrict__ O,    // [N,d]  (not used in main loop; only D needed)
    const float* __restrict__ dO,   // [N,d]
    const float* __restrict__ L,    // [N]    (logsumexp from forward)
    const float* __restrict__ D,    // [N]    (rowsum(dO ⊙ O))
    float* __restrict__ dQ,         // [N,d]  (output)
    float* __restrict__ dK,         // [N,d]  (output, atomics)
    float* __restrict__ dV,         // [N,d]  (output, atomics)
    int N, int d,
    int Br, int Bc, int Tc,
    float scale)
{
    extern __shared__ float smem[];
    // Layout: [Qi Br*d][dOi Br*d][Li Br][Di Br][Kj Bc*d][Vj Bc*d]
    float* Qi  = smem;
    float* dOi = Qi + Br*d;
    float* Li  = dOi + Br*d;
    float* Di  = Li + Br;
    float* Kj  = Di + Br;
    float* Vj  = Kj + Bc*d;

    const int row_start = blockIdx.x * Br;
    const int r_local   = threadIdx.x;
    const int row       = row_start + r_local;

    int Br_tail = N - row_start;
    int Br_eff  = Br_tail < 0 ? 0 : (Br_tail < Br ? Br_tail : Br);
    if (r_local >= Br_eff || row >= N) return;

    // Load Qi, dOi, Li, Di; init dQi local buffer to 0
    for (int t=0; t<d; ++t){
        Qi [r_local*d + t] = Q [row*d + t];
        dOi[r_local*d + t] = dO[row*d + t];
    }
    Li[r_local] = L[row];
    Di[r_local] = D[row];

    // Iterate K/V tiles
    for (int j=0; j<Tc; ++j){
        const int col_start = j * Bc;
        int Bc_tail = N - col_start;
        int Bc_eff  = Bc_tail < 0 ? 0 : (Bc_tail < Bc ? Bc_tail : Bc);

        // Load Kj, Vj cooperatively
        for (int tt=r_local; tt<Bc_eff*d; tt += Br_eff){
            int r = tt / d, c = tt % d;
            Kj[r*d + c] = K[(col_start + r)*d + c];
            Vj[r*d + c] = V[(col_start + r)*d + c];
        }
        __syncthreads();

        // --- Main math per row r (split-Q) ---
        const float* qrow = &Qi[r_local*d];
        const float* dorow= &dOi[r_local*d];
        const float Li_row = Li[r_local];
        const float Di_row = Di[r_local];

        extern __shared__ float extra[];

        // Allocate a temp dQ_row in registers via dynamic new is not allowed.
        // We do a two-pass pattern per feature t: compute ds_rc then update dQ[row,t] immediately.
        for (int t=0; t<d; ++t){
            float dQ_acc_t = 0.f;

            // Sweep columns c
            for (int c=0; c<Bc_eff; ++c){
                // s = (Qi_r · Kj_c) * scale
                const float* kptr = &Kj[c*d];
                float s=0.f; for (int u=0; u<d; ++u) s += qrow[u]*kptr[u];
#if USE_SCALE
                s *= scale;
#endif
                float P_rc = __expf(s - Li_row);

                // dp_rc = dO_r · Vj_c
                const float* vptr = &Vj[c*d];
                float dp=0.f; for (int u=0; u<d; ++u) dp += dorow[u]*vptr[u];

                float ds = P_rc * (dp - Di_row);

                // accumulate dQ[row,t]
                dQ_acc_t += ds * kptr[t];

                // accumulate dK[c,t] (atomic)
                atomicAdd(&dK[(col_start + c)*d + t], ds * qrow[t]);

                // accumulate dV[c,t] (atomic)
                atomicAdd(&dV[(col_start + c)*d + t], P_rc * dorow[t]);
            }
            // write dQ once for this feature
            atomicAdd(&dQ[row*d + t], dQ_acc_t);
        }
        __syncthreads();
    }
}

// -------------------------- Host wrapper & demo -------------------------------
void fa2_backward_cuda(const std::vector<float>& hQ,
                       const std::vector<float>& hK,
                       const std::vector<float>& hV,
                       const std::vector<float>& hO,
                       const std::vector<float>& h_dO,
                       const std::vector<float>& hL,
                       int N,int d,
                       std::vector<float>& h_dQ,
                       std::vector<float>& h_dK,
                       std::vector<float>& h_dV)
{
    size_t bytes = (size_t)N*d*sizeof(float);

    float *Q,*K,*V,*O,*dO,*L,*D,*dQ,*dK,*dV;
    CHECK_CUDA(cudaMalloc(&Q, bytes));
    CHECK_CUDA(cudaMalloc(&K, bytes));
    CHECK_CUDA(cudaMalloc(&V, bytes));
    CHECK_CUDA(cudaMalloc(&O, bytes));
    CHECK_CUDA(cudaMalloc(&dO, bytes));
    CHECK_CUDA(cudaMalloc(&L, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&D, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dQ, bytes));
    CHECK_CUDA(cudaMalloc(&dK, bytes));
    CHECK_CUDA(cudaMalloc(&dV, bytes));

    CHECK_CUDA(cudaMemcpy(Q, hQ.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(K, hK.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(V, hV.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(O, hO.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dO, h_dO.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(L, hL.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dQ, 0, bytes));
    CHECK_CUDA(cudaMemset(dK, 0, bytes));
    CHECK_CUDA(cudaMemset(dV, 0, bytes));

    // D = rowsum(dO ⊙ O)
    int tb = 128, gb = (N + tb - 1) / tb;
    rowsum_dot_kernel<<<gb, tb>>>(dO, O, D, N, d);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Tile choices (simple defaults; tune per device)
    int Br = std::min(N, 64);
    int Bc = std::min(N, 64);
    int Tr = (N + Br - 1)/Br;
    int Tc = (N + Bc - 1)/Bc;

    dim3 grid(Tr,1,1);
    dim3 block(Br,1,1);

    size_t shmem = sizeof(float)*((2*Br + 2*Bc)*d + 2*Br); // Qi+dOi + Li+Di + Kj+Vj
    CHECK_CUDA(cudaFuncSetAttribute(fa2_backward_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem));

    float scale = 1.0f/std::sqrt((float)d);
#if !USE_SCALE
    scale = 1.0f;
#endif

    fa2_backward_kernel<<<grid, block, shmem>>>(
        Q,K,V,O,dO,L,D, dQ,dK,dV, N,d, Br,Bc,Tc, scale);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    h_dQ.resize(N*d); h_dK.resize(N*d); h_dV.resize(N*d);
    CHECK_CUDA(cudaMemcpy(h_dQ.data(), dQ, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dK.data(), dK, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dV.data(), dV, bytes, cudaMemcpyDeviceToHost));

    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O); cudaFree(dO);
    cudaFree(L); cudaFree(D); cudaFree(dQ); cudaFree(dK); cudaFree(dV);
}

int main(){
    // Tiny sanity example: reuse the 2x4 case.
    int N=2, d=4;
    std::vector<float> Q = {1,0,1,0,  0,1,0,1};
    std::vector<float> K = {1,0,1,0,  0,1,0,1};
    std::vector<float> V = {10,20,30,40,  50,60,70,80};
    // Forward outputs 
    std::vector<float> O = {
        20.75766f, 30.75766f, 40.75766f, 50.75766f,
        39.24234f, 49.24234f, 59.24234f, 69.24235f };
    std::vector<float> L = {1.3132621f, 1.3132621f};

    // Provide a test dO; for demo, use ones
    std::vector<float> dO(N*d, 1.0f);

    std::vector<float> dQ, dK, dV;
    fa2_backward_cuda(Q,K,V,O,dO,L, N,d, dQ,dK,dV);

    printf("FA-2 backward, USE_SCALE=%d\n", USE_SCALE);
    for (int i=0;i<N;i++){
        printf("dQ[%d]: ", i);
        for (int t=0;t<d;t++) printf("%.5f ", dQ[i*d+t]);
        printf("\n");
    }
    for (int i=0;i<N;i++){
        printf("dK[%d]: ", i);
        for (int t=0;t<d;t++) printf("%.5f ", dK[i*d+t]);
        printf("\n");
    }
    for (int i=0;i<N;i++){
        printf("dV[%d]: ", i);
        for (int t=0;t<d;t++) printf("%.5f ", dV[i*d+t]);
        printf("\n");
    }
    return 0;
}