
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

// ========================= CUDA kernel (FA-2 forward) =====================================
__global__ void fa2_forward_kernel(
    const float* __restrict__ Q,   // [N,d]
    const float* __restrict__ K,   // [N,d]
    const float* __restrict__ V,   // [N,d]
    float* __restrict__ O,         // [N,d]
    float* __restrict__ L,         // [N]
    int N, int d,
    int Br, int Bc, int Tc,
    float scale)
{
    extern __shared__ float smem[];
    // Shared layout: [Qi Br*d][Otilde Br*d][li Br][mi Br][Kj Bc*d][Vj Bc*d][tmax Br]
    float* Qi   = smem;
    float* Oi   = Qi + Br*d;       // running unnormalized O~
    float* li   = Oi + Br*d;       // running sum of exp
    float* mi   = li + Br;         // running row max
    float* Kj   = mi + Br;
    float* Vj   = Kj + Bc*d;
    float* tmax = Vj + Bc*d;

    const int row_start = blockIdx.x * Br;
    const int r_local   = threadIdx.x;
    const int row       = row_start + r_local;

    int Br_tail = N - row_start;
    int Br_eff  = Br_tail < 0 ? 0 : (Br_tail < Br ? Br_tail : Br);
    if (r_local >= Br_eff || row >= N) return;

    // Load Qi and init O~, ell, m
    for (int c=0;c<d;++c){
        Qi[r_local*d + c] = Q[row*d + c];
        Oi[r_local*d + c] = 0.f;
    }
    li[r_local] = 0.f;
    mi[r_local] = -CUDART_INF_F;
    __syncthreads();

    // Sweep over KV tiles
    for (int j=0;j<Tc;++j){
        const int col_start = j * Bc;
        int Bc_tail = N - col_start;
        int Bc_eff  = Bc_tail < 0 ? 0 : (Bc_tail < Bc ? Bc_tail : Bc);

        // Load K_j, V_j cooperatively (stride by Br_eff threads)
        for (int t = r_local; t < Bc_eff*d; t += Br_eff){
            int r = t / d, c = t % d;
            Kj[r*d + c] = K[(col_start + r)*d + c];
            Vj[r*d + c] = V[(col_start + r)*d + c];
        }
        __syncthreads();

        // Pass 1: rowwise max for current tile
        float my_max = -CUDART_INF_F;
        for (int c=0;c<Bc_eff;++c){
            const float* qptr = &Qi[r_local*d];
            const float* kptr = &Kj[c*d];
            float dot=0.f;
            #pragma unroll 1
            for (int t=0;t<d;++t) dot += qptr[t]*kptr[t];
#if USE_SCALE
            float s = dot * scale;
#else
            float s = dot;
#endif
            if (s > my_max) my_max = s;
        }
        tmax[r_local] = my_max;
        __syncthreads();

        // Pass 2: l_t = sum exp(s - m_t)
        float m_t = tmax[r_local];
        float l_t = 0.f;
        for (int c=0;c<Bc_eff;++c){
            const float* qptr = &Qi[r_local*d];
            const float* kptr = &Kj[c*d];
            float dot=0.f;
            #pragma unroll 1
            for (int t=0;t<d;++t) dot += qptr[t]*kptr[t];
#if USE_SCALE
            float s = dot * scale;
#else
            float s = dot;
#endif
            l_t += __expf(s - m_t);
        }
        __syncthreads();

        // Online state update (FA-2)
        float m_old = mi[r_local];
        float l_old = li[r_local];
        float m_new = fmaxf(m_old, m_t);
        float l_new = __expf(m_old - m_new) * l_old + l_t;

        // Rescale previous O~ 
        if (l_old > 0.f) {
            float scale_up = __expf(m_new - m_old);
            for (int t=0;t<d;++t)
            Oi[r_local*d + t] *= scale_up;

        }
        
        // Add unnormalized PV for this tile
        for (int t=0;t<d;++t){
            float acc_un = 0.f;
            for (int c=0;c<Bc_eff;++c){
                const float* qptr = &Qi[r_local*d];
                const float* kptr = &Kj[c*d];
                float dot=0.f;
                #pragma unroll 1
                for (int tt=0; tt<d; ++tt) dot += qptr[tt]*kptr[tt];
#if USE_SCALE
                float s = dot * scale;
#else
                float s = dot;
#endif
                acc_un += __expf(s - m_t) * Vj[c*d + t];
            }
            Oi[r_local*d + t] += acc_un;
        }

        // Commit (m, ell)
        mi[r_local] = m_new;
        li[r_local] = l_new;
        __syncthreads();
    }

    // Final normalize once, and write L = m + log(ell)
    float ell  = li[r_local];
    float mfin = mi[r_local];
    float inv_ell = (ell > 0.f) ? (1.0f/ell) : 0.f;
    for (int t=0;t<d;++t)
        O[row*d + t] = Oi[r_local*d + t] * inv_ell;
    L[row] = mfin + logf(fmaxf(ell, 1e-38f));
}

// ========================= Host launcher ======================================
void fa2_forward_cuda(const std::vector<float>& hQ,
                      const std::vector<float>& hK,
                      const std::vector<float>& hV,
                      int N,int d,
                      std::vector<float>& hO,
                      std::vector<float>& hL)
{
    size_t bytes = (size_t)N*d*sizeof(float);
    float *Q,*K,*V,*O,*L;
    CHECK_CUDA(cudaMalloc(&Q, bytes));
    CHECK_CUDA(cudaMalloc(&K, bytes));
    CHECK_CUDA(cudaMalloc(&V, bytes));
    CHECK_CUDA(cudaMalloc(&O, bytes));
    CHECK_CUDA(cudaMalloc(&L, N*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(Q, hQ.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(K, hK.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(V, hV.data(), bytes, cudaMemcpyHostToDevice));

    // Tiles (tune per device)
    int Br = std::min(N, 64);
    int Bc = std::min(N, 64);
    int Tr = (N + Br - 1)/Br;
    int Tc = (N + Bc - 1)/Bc;

    dim3 grid(Tr,1,1);
    dim3 block(Br,1,1);

    size_t shmem = sizeof(float)*((2*Br + 2*Bc)*d + 3*Br);
    CHECK_CUDA(cudaFuncSetAttribute(fa2_forward_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem));

    float scale = 1.0f/std::sqrt((float)d);
#if !USE_SCALE
    scale = 1.0f;
#endif

    fa2_forward_kernel<<<grid, block, shmem>>>(Q,K,V,O,L, N,d, Br,Bc,Tc, scale);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    hO.resize(N*d); hL.resize(N);
    CHECK_CUDA(cudaMemcpy(hO.data(), O, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hL.data(), L, N*sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O); cudaFree(L);
}

// ========================= Demo ==============================================
int main(){
    int N=2, d=4;
    std::vector<float> Q = {1,0,1,0,  0,1,0,1};
    std::vector<float> K = {1,0,1,0,  0,1,0,1};
    std::vector<float> V = {10,20,30,40,  50,60,70,80};

    std::vector<float> O, L;
    fa2_forward_cuda(Q,K,V, N,d, O, L);

    printf("FA-2 forward, USE_SCALE=%d\n", USE_SCALE);
    for (int i=0;i<N;i++){
        printf("O[%d]: ", i);
        for (int t=0;t<d;t++) printf("%.5f ", O[i*d+t]);
        printf(" | L=%.6f\n", L[i]);
    }
    return 0;
}
