#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <math_constants.h>

#define CHECK_CUDA(x) do { auto e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#ifndef USE_SCALE
#define USE_SCALE 1   
#endif

// ========================= Forward kernel (CTA per Qi) =========================
__global__ void fa2_forward_kernel(
    const float* __restrict__ Q, // [N,d]
    const float* __restrict__ K, // [N,d]
    const float* __restrict__ V, // [N,d]
    float* __restrict__ O,       // [N,d]
    float* __restrict__ L,       // [N]
    int N, int d, int Br, int Bc, int Tc, float scale)
{
    extern __shared__ float sm[];
    // [Qi Br*d][Otilde Br*d][li Br][mi Br][Kj Bc*d][Vj Bc*d]
    float* Qi   = sm;
    float* Oi   = Qi + Br*d;              
    float* li   = Oi + Br*d;              
    float* mi   = li + Br;                
    float* Kj   = mi + Br;
    float* Vj   = Kj + Bc*d;           

    const int row0 = blockIdx.x * Br;
    const int rloc = threadIdx.x;
    const int row  = row0 + rloc;

    const int Br_eff = max(0, min(Br, N - row0));
    if (rloc >= Br_eff || row >= N) return;

    // Load Qi, init O~, l, m
    for (int c=0;c<d;++c){ Qi[rloc*d+c] = Q[row*d+c]; Oi[rloc*d+c] = 0.f; }
    li[rloc] = 0.f; mi[rloc] = -CUDART_INF_F;
    __syncthreads();

    // Loop over KV tiles
    for (int j=0;j<Tc;++j){
        const int col0 = j * Bc;
        const int Bc_eff = max(0, min(Bc, N - col0));

        // Load Kj, Vj cooperatively
        for (int t = rloc; t < Bc_eff*d; t += Br_eff){
            int r = t / d, c = t % d;
            Kj[r*d+c] = K[(col0+r)*d + c];
            Vj[r*d+c] = V[(col0+r)*d + c];
        }
        __syncthreads();

        // Pass1: m_t = rowwise max of scaled logits on this tile
        float m_t = -CUDART_INF_F;
        const float* qrow = &Qi[rloc*d];
        for (int c=0;c<Bc_eff;++c){
            const float* kptr = &Kj[c*d];
            float dot=0.f; 
            #pragma unroll 1
            for (int t=0;t<d;++t) dot += qrow[t]*kptr[t];
#if USE_SCALE
            dot *= scale;
#endif
            m_t = fmaxf(m_t, dot);
        }
        __syncthreads();

        // Pass2: l_t = sum exp(s - m_t)
        float l_t = 0.f;
        for (int c=0;c<Bc_eff;++c){
            const float* kptr = &Kj[c*d];
            float dot=0.f; 
            #pragma unroll 1
            for (int t=0;t<d;++t) dot += qrow[t]*kptr[t];
#if USE_SCALE
            dot *= scale;
#endif
            l_t += __expf(dot - m_t);
        }

        // Online state update (FA-2 style)
        float m_old = mi[rloc], l_old = li[rloc];
        float m_new = fmaxf(m_old, m_t);
        float l_new = __expf(m_old - m_new)*l_old + l_t;

        // Rescale previous O~ only if there was previous contribution
        if (l_old > 0.f){
            float up = __expf(m_new - m_old);
            for (int t=0;t<d;++t) Oi[rloc*d+t] *= up;
        }

        // Accumulate current tile's unnormalized PV into O~
        for (int t=0;t<d;++t){
            float acc=0.f;
            for (int c=0;c<Bc_eff;++c){
                const float* kptr = &Kj[c*d];
                const float* vptr = &Vj[c*d];
                float dot=0.f; 
                #pragma unroll 1
                for (int u=0;u<d;++u) dot += qrow[u]*kptr[u];
#if USE_SCALE
                dot *= scale;
#endif
                acc += __expf(dot - m_t)*vptr[t];
            }
            Oi[rloc*d+t] += acc;
        }

        mi[rloc] = m_new; li[rloc] = l_new;
        __syncthreads();
    }

    // Final normalize once: O = O~/ell ; L = m + log(ell)
    float ell = li[rloc];
    float inv = (ell>0.f) ? 1.f/ell : 0.f;
    for (int t=0;t<d;++t) O[row*d+t] = Oi[rloc*d+t]*inv;
    L[row] = mi[rloc] + logf(fmaxf(ell, 1e-38f));
}

// =================== Backward helpers: D = rowsum(dO ⊙ O) =====================
__global__ void rowsum_dot_kernel(const float* __restrict__ dO,
                                  const float* __restrict__ O,
                                  float* __restrict__ D,
                                  int N,int d){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=N) return;
    const float* a = dO + i*d;
    const float* b =  O + i*d;
    float s=0.f; for (int t=0;t<d;++t) s += a[t]*b[t];
    D[i]=s;
}

// ================ Backward kernel (CTA per Kj,Vj column-tile) =================
// Each CTA:
//  - loads one (Kj,Vj) tile and keeps local dKj,dVj in shared
//  - loops over ALL Qi tiles (i = 0..Tr-1): recomputes logits, builds P_ij and
//    the two matmuls needed, does atomicAdd to dQ rows
//  - at the end, writes dKj,dVj once to HBM
__global__ void fa2_backward_kernel(
    const float* __restrict__ Q,   // [N,d]
    const float* __restrict__ K,   // [N,d]
    const float* __restrict__ V,   // [N,d]
    const float* __restrict__ O,   // [N,d] (not directly used; only D matters)
    const float* __restrict__ dO,  // [N,d]
    const float* __restrict__ L,   // [N]
    const float* __restrict__ D,   // [N] = rowsum(dO ⊙ O)
    float* __restrict__ dQ,        // [N,d] (atomicAdd)
    float* __restrict__ dK,        // [N,d] (single-write per CTA tile)
    float* __restrict__ dV,        // [N,d] (single-write per CTA tile)
    int N,int d, int Br,int Bc,int Tr,int Tc, float scale)
{
    extern __shared__ float sm[];
    // Shared: [Kj Bc*d][Vj Bc*d][dKj Bc*d][dVj Bc*d][Qi Br*d][dOi Br*d][Li Br][Di Br]
    float* Kj  = sm;
    float* Vj  = Kj  + Bc*d;
    float* dKj = Vj  + Bc*d;
    float* dVj = dKj + Bc*d;
    float* Qi  = dVj + Bc*d;
    float* dOi = Qi  + Br*d;
    float* Li  = dOi + Br*d;
    float* Di  = Li  + Br;

    const int j = blockIdx.x;               // CTA id = column tile id
    const int col0 = j * Bc;
    const int Bc_eff = max(0, min(Bc, N - col0));
    if (Bc_eff <= 0) return;

    // Load Kj,Vj once and init dKj,dVj to 0
    for (int t = threadIdx.x; t < Bc_eff*d; t += blockDim.x){
        int r=t/d, c=t%d;
        Kj[r*d+c] = K[(col0+r)*d + c];
        Vj[r*d+c] = V[(col0+r)*d + c];
        dKj[r*d+c]=0.f;
        dVj[r*d+c]=0.f;
    }
    __syncthreads();

    // Iterate ALL row tiles (Qi blocks)
    for (int i=0;i<Tr;++i){
        const int row0 = i * Br;
        const int Br_eff = max(0, min(Br, N - row0));
        if (Br_eff<=0) continue;

        // Load Qi,dOi,Li,Di
        for (int t = threadIdx.x; t < Br_eff*d; t += blockDim.x){
            int r=t/d, c=t%d;
            Qi[r*d+c]  = Q [(row0+r)*d + c];
            dOi[r*d+c] = dO[(row0+r)*d + c];
        }
        for (int r = threadIdx.x; r < Br_eff; r += blockDim.x){
            Li[r] = L[row0+r];
            Di[r] = D[row0+r];
        }
        __syncthreads();

        // For each row in Qi, compute contributions with this (Kj,Vj) tile
        for (int r = threadIdx.x; r < Br_eff; r += blockDim.x){
            const float* qrow  = &Qi[r*d];
            const float Lr = Li[r];
            const float Dr = Di[r];

            for (int c=0;c<Bc_eff;++c){
                const float* kptr = &Kj[c*d];
                const float* vptr = &Vj[c*d];

                // logits
                float s=0.f; for (int u=0;u<d;++u) s += qrow[u]*kptr[u];
#if USE_SCALE
                s *= scale;
#endif
                float P = __expf(s - Lr);

                // dp = dOi · Vj[c,:]
                float dp=0.f; for (int u=0;u<d;++u) dp += dOi[r*d+u]*vptr[u];

                float ds = P * (dp - Dr);

                // dQ[row0+r, :] += ds * Kj[c,:]
                for (int t=0;t<d;++t){
                    atomicAdd(&dQ[(row0+r)*d + t], ds * kptr[t]);
                }

                // Accumulate dKj[c,:] += ds * Qi[r,:]
                for (int t=0;t<d;++t){
                    dKj[c*d + t] += ds * qrow[t];
                }

                // Accumulate dVj[c,:] += P * dOi[r,:]
                for (int t=0;t<d;++t){
                    dVj[c*d + t] += P * dOi[r*d + t];
                }
            }
        }
        __syncthreads();
    }

    // Write back dKj, dVj once 
    for (int t = threadIdx.x; t < Bc_eff*d; t += blockDim.x){
        int r=t/d, c=t%d;
        dK[(col0+r)*d + c] = dKj[r*d + c]; 
        dV[(col0+r)*d + c] = dVj[r*d + c];
    }
}

// ============================= Host wrappers ==================================
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

    // Tile sizes (simple defaults; tune per device)
    int Br = std::min(N, 64);
    int Bc = std::min(N, 64);
    int Tr = (N + Br - 1)/Br;
    int Tc = (N + Bc - 1)/Bc;

    dim3 gridF(Tr,1,1);
    dim3 blockF(Br,1,1);
    size_t shF = sizeof(float)*((2*Br + 2*Bc)*d + 2*Br + 0*Bc); // forward layout above

    CHECK_CUDA(cudaFuncSetAttribute(fa2_forward_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shF));

    float scale = 1.0f/std::sqrt((float)d);
#if !USE_SCALE
    scale = 1.0f;
#endif
    fa2_forward_kernel<<<gridF, blockF, shF>>>(Q,K,V,O,L, N,d, Br,Bc,Tc, scale);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    hO.resize(N*d); hL.resize(N);
    CHECK_CUDA(cudaMemcpy(hO.data(), O, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hL.data(), L, N*sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O); cudaFree(L);
}

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
    int tb=128, gb=(N+tb-1)/tb;
    rowsum_dot_kernel<<<gb,tb>>>(dO,O,D,N,d);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Tiles and launch (CTA per column tile)
    int Br = std::min(N, 64);
    int Bc = std::min(N, 64);
    int Tr = (N + Br - 1)/Br;
    int Tc = (N + Bc - 1)/Bc;

    dim3 gridB(Tc,1,1);
    // Threads: use max(Br,Bc) but keep <= 256 for simplicity
    int tpb = std::max(Br,Bc);
    tpb = std::min(tpb, 256);
    dim3 blockB(tpb,1,1);

    size_t shB = sizeof(float)*( (4*Bc + 2*Br)*d + 2*Br );
    CHECK_CUDA(cudaFuncSetAttribute(fa2_backward_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shB));

    float scale = 1.0f/std::sqrt((float)d);
#if !USE_SCALE
    scale = 1.0f;
#endif

    fa2_backward_kernel<<<gridB, blockB, shB>>>(
        Q,K,V,O,dO,L,D, dQ,dK,dV, N,d, Br,Bc,Tr,Tc, scale);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    h_dQ.resize(N*d); h_dK.resize(N*d); h_dV.resize(N*d);
    CHECK_CUDA(cudaMemcpy(h_dQ.data(), dQ, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dK.data(), dK, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dV.data(), dV, bytes, cudaMemcpyDeviceToHost));

    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O); cudaFree(dO);
    cudaFree(L); cudaFree(D); cudaFree(dQ); cudaFree(dK); cudaFree(dV);
}

// =============================== Demo =========================================
int main(){
    int N=2, d=4;
    std::vector<float> Q = {1,0,1,0,  0,1,0,1};
    std::vector<float> K = {1,0,1,0,  0,1,0,1};
    std::vector<float> V = {10,20,30,40,  50,60,70,80};

    // Forward
    std::vector<float> O,L;
    fa2_forward_cuda(Q,K,V, N,d, O,L);

    printf("Forward (FA-2), USE_SCALE=%d\n", USE_SCALE);
    for (int i=0;i<N;i++){
        printf("O[%d]: ",i);
        for (int t=0;t<d;t++) printf("%.5f ", O[i*d+t]);
        printf(" | L=%.6f\n", L[i]);
    }

    // Backward: simple dO = ones
    std::vector<float> dO(N*d, 1.0f), dQ,dK,dV;
    fa2_backward_cuda(Q,K,V,O,dO,L, N,d, dQ,dK,dV);

    printf("\nBackward (FA-2, §3.2 parallelism)\n");
    for (int i=0;i<N;i++){
        printf("dQ[%d]: ",i);
        for (int t=0;t<d;t++) printf("%.5f ", dQ[i*d+t]); printf("\n");
    }
    for (int i=0;i<N;i++){
        printf("dK[%d]: ",i);
        for (int t=0;t<d;t++) printf("%.5f ", dK[i*d+t]); printf("\n");
    }
    for (int i=0;i<N;i++){
        printf("dV[%d]: ",i);
        for (int t=0;t<d;t++) printf("%.5f ", dV[i*d+t]); printf("\n");
    }
    return 0;
}