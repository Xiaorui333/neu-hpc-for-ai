
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>
#include <math_constants.h>

#define CHECK_CUDA(x) do { \
  cudaError_t e=(x); \
  if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
    exit(1);} \
} while(0)

#ifndef USE_SCALE
#define USE_SCALE 1
#endif

// ---------------- warp helpers ----------------
__inline__ __device__ float warp_sum(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v; // sum exists in lane 0 (others hold partials)
}

__inline__ __device__ int ceil_div(int a, int b){ return (a + b - 1) / b; }

// ========================= Forward (CTA per Qi; warp over d; split-Q) ===================
__global__ void fa2_forward_kernel_warp(
    const float* __restrict__ Q, // [N,d]
    const float* __restrict__ K, // [N,d]
    const float* __restrict__ V, // [N,d]
    float* __restrict__ O,       // [N,d]
    float* __restrict__ L,       // [N]
    int N, int d, int Br, int Bc, int Tc, float scale)
{
    extern __shared__ float sm[];
    // Layout:
    // [Qi Br*d][Otilde Br*d][li Br][mi Br][Kj Bc*d][Vj Bc*d][scores W*Bc]
    float* Qi   = sm;
    float* Oi   = Qi + Br*d;
    float* li   = Oi + Br*d;
    float* mi   = li + Br;
    float* Kj   = mi + Br;
    float* Vj   = Kj + Bc*d;
    // per-warp score cache (for current Kj tile): s_buf[c] = q·k_c * scale
    float* ScoresAll = Vj + Bc*d;

    const int row0 = blockIdx.x * Br;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;          // warp id in block
    const int lane = tid & 31;          // lane id in warp
    const int W    = blockDim.x >> 5;   // warps per block

    const int Br_eff = max(0, min(Br, N - row0));
    if (Br_eff <= 0) return;

    // cooperative load Qi; init O~, l, m
    for (int t = tid; t < Br_eff*d; t += blockDim.x){
        int r = t / d, c = t % d;
        Qi[r*d + c] = Q[(row0 + r)*d + c];
        Oi[r*d + c] = 0.f;
    }
    for (int r = tid; r < Br_eff; r += blockDim.x){
        li[r] = 0.f; mi[r] = -CUDART_INF_F;
    }
    __syncthreads();

    // Each warp owns a subset of rows 
    int rows_per_warp = ceil_div(Br_eff, W);

    for (int j=0; j<Tc; ++j){
        const int col0 = j * Bc;
        const int Bc_eff = max(0, min(Bc, N - col0));
        if (Bc_eff <= 0) break;

        // load Kj,Vj
        for (int t = tid; t < Bc_eff*d; t += blockDim.x){
            int r = t / d, c = t % d;
            Kj[r*d + c] = K[(col0 + r)*d + c];
            Vj[r*d + c] = V[(col0 + r)*d + c];
        }
        __syncthreads();

        // Per-warp slice of the score cache
        float* s_buf = ScoresAll + warp * Bc;

        // Each warp processes its assigned rows
        for (int rr = 0; rr < rows_per_warp; ++rr){
            int r = warp * rows_per_warp + rr;
            if (r >= Br_eff) break;

            const float* qrow = &Qi[r*d];

            // pass1: compute all scores s_c = q·k_c * scale, store in s_buf[c], compute m_t
            float m_t = -CUDART_INF_F;
            for (int c = 0; c < Bc_eff; ++c){
                const float* kptr = &Kj[c*d];
                float part = 0.f;
                for (int u = lane; u < d; u += 32) part += qrow[u] * kptr[u];
                float dot = warp_sum(part);              // lane 0 has sum
                if (lane == 0){
#if USE_SCALE
                    dot *= scale;
#endif
                    s_buf[c] = dot;                      // cache the scaled score
                    m_t = fmaxf(m_t, dot);
                }
            }
            // broadcast m_t
            m_t = __shfl_sync(0xffffffffu, m_t, 0);

            // pass2: l_t = Σ_c exp(s_c - m_t), computed by lane 0 then broadcast scalars
            float l_t = 0.f;
            if (lane == 0){
                for (int c = 0; c < Bc_eff; ++c) l_t += __expf(s_buf[c] - m_t);
            }
            l_t = __shfl_sync(0xffffffffu, l_t, 0);

            // online state update with correct rebasing (alpha/beta)
            float m_old = mi[r], l_old = li[r];
            float m_new = fmaxf(m_old, m_t);
            float alpha = __expf(m_old - m_new);  
            float beta  = __expf(m_t   - m_new);  
            float l_new = alpha * l_old + beta * l_t;

            // broadcast alpha/beta for lanes
            alpha = __shfl_sync(0xffffffffu, alpha, 0);
            beta  = __shfl_sync(0xffffffffu, beta , 0);

            // rescale previous O~ by alpha
            for (int t = lane; t < d; t += 32) Oi[r*d + t] *= alpha;

            // accumulate current tile contribution with beta
            for (int t = lane; t < d; t += 32){
                float acc = 0.f;
                for (int c = 0; c < Bc_eff; ++c){
                    const float* vptr = &Vj[c*d];
                    float w_un = __expf(s_buf[c] - m_t);
                    acc += w_un * vptr[t];
                }
                Oi[r*d + t] += beta * acc;
            }

            if (lane == 0){ mi[r] = m_new; li[r] = l_new; }
        }
        __syncthreads();
    }

    // final normalize (each warp writes its rows)
    for (int rr = 0; rr < rows_per_warp; ++rr){
        int r = warp * rows_per_warp + rr;
        if (r >= Br_eff) break;
        float ell = li[r];
        float inv = (ell>0.f) ? 1.f/ell : 0.f;
        for (int t = lane; t < d; t += 32) O[(row0 + r)*d + t] = Oi[r*d + t] * inv;
        if (lane == 0) L[row0 + r] = mi[r] + logf(fmaxf(ell, 1e-38f));
    }
}

// ============ Backward helpers: D = rowsum(dO ⊙ O) (row dot) ==================
__global__ void rowsum_dot_kernel(const float* __restrict__ dO,
                                  const float* __restrict__ O,
                                  float* __restrict__ D,
                                  int N,int d){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    const float* a = dO + row*d;
    const float* b =  O + row*d;
    float acc = 0.f;
    for (int t=0; t<d; ++t) acc += a[t]*b[t];
    D[row] = acc;
}

// ===== Backward (CTA per Kj,Vj; warp over d; dKj/dVj kept on-chip; dQ atomic) ==
__global__ void fa2_backward_kernel_warp(
    const float* __restrict__ Q,   // [N,d]
    const float* __restrict__ K,   // [N,d]
    const float* __restrict__ V,   // [N,d]
    const float* __restrict__ dO,  // [N,d]
    const float* __restrict__ L,   // [N]
    const float* __restrict__ D,   // [N]
    float* __restrict__ dQ,        // [N,d]  
    float* __restrict__ dK,        // [N,d]  
    float* __restrict__ dV,        // [N,d]
    int N,int d, int Br,int Bc,int Tr, float scale)
{
    extern __shared__ float sm[];
    // [Kj Bc*d][Vj Bc*d][dKj Bc*d][dVj Bc*d][Qi Br*d][dOi Br*d][Li Br][Di Br]
    float* Kj  = sm;
    float* Vj  = Kj  + Bc*d;
    float* dKj = Vj  + Bc*d;
    float* dVj = dKj + Bc*d;
    float* Qi  = dVj + Bc*d;
    float* dOi = Qi  + Br*d;
    float* Li  = dOi + Br*d;
    float* Di  = Li  + Br;

    const int j = blockIdx.x;           // CTA processes (Kj,Vj)
    const int col0 = j * Bc;
    const int Bc_eff = max(0, min(Bc, N - col0));
    if (Bc_eff <= 0) return;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int W    = blockDim.x >> 5;
    (void)warp; (void)W;

    // load Kj,Vj; zero dKj,dVj
    for (int t = tid; t < Bc_eff*d; t += blockDim.x){
        int r=t/d, c=t%d;
        Kj[r*d+c] = K[(col0 + r)*d + c];
        Vj[r*d+c] = V[(col0 + r)*d + c];
        dKj[r*d+c]= 0.f;
        dVj[r*d+c]= 0.f;
    }
    __syncthreads();

    // iterate all Qi tiles
    for (int i=0; i<Tr; ++i){
        const int row0 = i * Br;
        const int Br_eff = max(0, min(Br, N - row0));
        if (Br_eff <= 0) continue;

        // load Qi, dOi, Li, Di
        for (int t = tid; t < Br_eff*d; t += blockDim.x){
            int r=t/d, c=t%d;
            Qi [r*d+c] = Q [(row0 + r)*d + c];
            dOi[r*d+c] = dO[(row0 + r)*d + c];
        }
        for (int r = tid; r < Br_eff; r += blockDim.x){
            Li[r] = L[row0 + r];
            Di[r] = D[row0 + r];
        }
        __syncthreads();

        int rows_per_warp = ceil_div(Br_eff, (blockDim.x>>5));
        for (int rr=0; rr<rows_per_warp; ++rr){
            int r = (tid>>5) * rows_per_warp + rr; // warp*rows_per_warp + rr
            if (r >= Br_eff) break;

            const float* qrow  = &Qi[r*d];
            const float* dorow = &dOi[r*d];
            const float  Lr    = Li[r];
            const float  Dr    = Di[r];

            // For each column c in this (Kj,Vj) tile:
            for (int c=0; c<Bc_eff; ++c){
                const float* kptr = &Kj[c*d];
                const float* vptr = &Vj[c*d];

                // s = q·k /√d  (warp-parallel over d)
                float part = 0.f;
                for (int u = lane; u < d; u += 32) part += qrow[u] * kptr[u];
                float s = warp_sum(part);
                s = __shfl_sync(0xffffffffu, s, 0);
#if USE_SCALE
                if (lane==0) s *= scale;
                s = __shfl_sync(0xffffffffu, s, 0);
#endif
                float P = __expf(s - Lr);

                // dp = dOi · v_c
                float dp_part = 0.f;
                for (int u = lane; u < d; u += 32) dp_part += dorow[u] * vptr[u];
                float dp = warp_sum(dp_part);

                // ds = P * (dp - D_r)
                float ds = 0.f;
                if (lane==0) ds = P * (dp - Dr);
                ds = __shfl_sync(0xffffffffu, ds, 0);

                // dQ[row, t] += ds * kptr[t]  (atomic across CTAs)
                for (int t = lane; t < d; t += 32)
                    atomicAdd(&dQ[(row0 + r)*d + t], ds * kptr[t]);

                // dKj[c, t] += ds * qrow[t]   (CTA-local, no atomics)
                for (int t = lane; t < d; t += 32)
                    dKj[c*d + t] += ds * qrow[t];

                // dVj[c, t] += P * dOi[row, t] (CTA-local)
                for (int t = lane; t < d; t += 32)
                    dVj[c*d + t] += P * dorow[t];
            }
        }
        __syncthreads();
    }

    // write back dKj,dVj once 
    for (int t = tid; t < Bc_eff*d; t += blockDim.x){
        int r=t/d, c=t%d;
        dK[(col0 + r)*d + c] = dKj[r*d + c];
        dV[(col0 + r)*d + c] = dVj[r*d + c];
    }
}

// ============================= Host helpers ===================================
static void cpu_attention_ref(const std::vector<float>& Q,
                              const std::vector<float>& K,
                              const std::vector<float>& V,
                              int N, int d,
                              std::vector<float>& O_ref,
                              std::vector<float>& L_ref)
{
    O_ref.assign(N*d, 0.f);
    L_ref.assign(N, 0.f);
    const float scale = USE_SCALE ? (1.0f/std::sqrt((float)d)) : 1.0f;

    std::vector<float> scores(N);
    for (int i=0;i<N;++i){
        // scores = Q[i]·K[:]
        float m = -INFINITY;
        for (int j=0;j<N;++j){
            float s=0.f;
            for (int t=0;t<d;++t) s += Q[i*d+t]*K[j*d+t];
            s *= scale;
            scores[j]=s; m = std::max(m,s);
        }
        // l = sum exp(s - m)
        double l=0.0;
        for (int j=0;j<N;++j) l += std::exp(scores[j]-m);
        double L = m + std::log(std::max(l, 1e-38));
        L_ref[i] = (float)L;
        // O = sum softmax * V
        for (int t=0;t<d;++t){
            double acc=0.0;
            for (int j=0;j<N;++j){
                double p = std::exp(scores[j]-L);
                acc += p * V[j*d+t];
            }
            O_ref[i*d+t] = (float)acc;
        }
    }
}

static void check_or_set_smem_attr(const void* func, size_t dynBytes){
    int dev=0; CHECK_CUDA(cudaGetDevice(&dev));
    int limit=0;
    CHECK_CUDA(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    if ((int)dynBytes > limit){
        fprintf(stderr,"[Warn] requested dynamic SMEM %zu exceeds opt-in limit %d; kernel may fail.\n", dynBytes, limit);
    }
    cudaError_t st = cudaFuncSetAttribute(func,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dynBytes);
    if (st != cudaSuccess){
        fprintf(stderr,"[Warn] cudaFuncSetAttribute failed: %s (requested %zu)\n",
                cudaGetErrorString(st), dynBytes);
    }
}

void fa2_forward_cuda(const std::vector<float>& hQ,
                      const std::vector<float>& hK,
                      const std::vector<float>& hV,
                      int N,int d,
                      std::vector<float>& hO,
                      std::vector<float>& hL,
                      int Br=64, int Bc=64, int W=4)
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

    Br = std::min(N, Br);
    Bc = std::min(N, Bc);
    int Tr = (N + Br - 1)/Br;
    int Tc = (N + Bc - 1)/Bc;

    dim3 gridF(Tr,1,1);
    dim3 blockF(W*32,1,1);

    // SMEM: (2*Br*d + 2*Br) + (2*Bc*d) + (W*Bc)
    size_t shF = sizeof(float) * ((2*Br + 2*Bc)*d + 2*Br + W*Bc);
    check_or_set_smem_attr((const void*)fa2_forward_kernel_warp, shF);

    float scale = 1.0f/std::sqrt((float)d);
#if !USE_SCALE
    scale = 1.0f;
#endif
    fa2_forward_kernel_warp<<<gridF, blockF, shF>>>(Q,K,V,O,L, N,d, Br,Bc,Tc, scale);
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
                       std::vector<float>& h_dV,
                       int Br=64, int Bc=64, int W=4)
{
    size_t bytes = (size_t)N*d*sizeof(float);
    float *Q,*K,*V,*dO,*L,*D,*dQ,*dK,*dV;
    CHECK_CUDA(cudaMalloc(&Q, bytes));
    CHECK_CUDA(cudaMalloc(&K, bytes));
    CHECK_CUDA(cudaMalloc(&V, bytes));
    CHECK_CUDA(cudaMalloc(&dO, bytes));
    CHECK_CUDA(cudaMalloc(&L, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&D, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dQ, bytes));
    CHECK_CUDA(cudaMalloc(&dK, bytes));
    CHECK_CUDA(cudaMalloc(&dV, bytes));

    CHECK_CUDA(cudaMemcpy(Q, hQ.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(K, hK.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(V, hV.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dO, h_dO.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(L, hL.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dQ, 0, bytes));
    CHECK_CUDA(cudaMemset(dK, 0, bytes));
    CHECK_CUDA(cudaMemset(dV, 0, bytes));

    // D = rowsum(dO ⊙ O)
    {
        float *O_dev;
        CHECK_CUDA(cudaMalloc(&O_dev, bytes));
        CHECK_CUDA(cudaMemcpy(O_dev, hO.data(), bytes, cudaMemcpyHostToDevice));
        int tb=128, gb=(N+tb-1)/tb;
        rowsum_dot_kernel<<<gb,tb>>>(dO, O_dev, D, N, d);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaFree(O_dev);
    }

    Br = std::min(N, Br);
    Bc = std::min(N, Bc);
    int Tr = (N + Br - 1)/Br;
    int Tc = (N + Bc - 1)/Bc;

    // CTA per column-tile
    dim3 gridB(Tc,1,1);
    dim3 blockB(W*32,1,1);
    // SMEM: (2*Bc*d) + (2*Bc*d) + (2*Br*d) + (2*Br)
    size_t shB = sizeof(float) * ((4*Bc + 2*Br)*d + 2*Br);
    check_or_set_smem_attr((const void*)fa2_backward_kernel_warp, shB);

    float scale = 1.0f/std::sqrt((float)d);
#if !USE_SCALE
    scale = 1.0f;
#endif

    fa2_backward_kernel_warp<<<gridB, blockB, shB>>>(
        Q,K,V,dO,L,D, dQ,dK,dV, N,d, Br,Bc,Tr, scale);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    h_dQ.resize(N*d); h_dK.resize(N*d); h_dV.resize(N*d);
    CHECK_CUDA(cudaMemcpy(h_dQ.data(), dQ, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dK.data(), dK, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_dV.data(), dV, bytes, cudaMemcpyDeviceToHost));

    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(dO); cudaFree(L);
    cudaFree(D); cudaFree(dQ); cudaFree(dK); cudaFree(dV);
}

// =============================== Demo / Tests =========================================
static void fill_random(std::vector<float>& v, float lo=-1.f, float hi=1.f, uint32_t seed=42){
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo,hi);
    for (auto& x: v) x = dist(rng);
}

int main(){
    // --- Sanity test (tiny) ---
    {
        int N=2, d=4;
        std::vector<float> Q = {1,0,1,0,  0,1,0,1};
        std::vector<float> K = {1,0,1,0,  0,1,0,1};
        std::vector<float> V = {10,20,30,40,  50,60,70,80};

        std::vector<float> O,L;
        fa2_forward_cuda(Q,K,V, N,d, O,L, /*Br*/64, /*Bc*/64, /*W*/4);

        printf("Forward (FA-2 warp), USE_SCALE=%d\n", USE_SCALE);
        for (int i=0;i<N;i++){
            printf("O[%d]: ",i);
            for (int t=0;t<d;t++) printf("%.5f ", O[i*d+t]);
            printf(" | L=%.6f\n", L[i]);
        }

        std::vector<float> dO(N*d, 1.0f), dQ,dK,dV;
        fa2_backward_cuda(Q,K,V,O,dO,L, N,d, dQ,dK,dV, /*Br*/64, /*Bc*/64, /*W*/4);

        printf("\nBackward (FA-2 warp, §3.3)\n");
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
    }

    // --- Random numerical check vs CPU reference ---
    {
        int N=64, d=64;
        std::vector<float> Q(N*d), K(N*d), V(N*d);
        fill_random(Q); fill_random(K, -0.5f, 0.5f, 123); fill_random(V, -1.f, 1.f, 999);

        std::vector<float> O,L;
        fa2_forward_cuda(Q,K,V, N,d, O,L, /*Br*/64, /*Bc*/64, /*W*/4);

        std::vector<float> O_ref, L_ref;
        cpu_attention_ref(Q,K,V, N,d, O_ref, L_ref);

        double max_abs_O=0.0, max_abs_L=0.0;
        for (int i=0;i<N*d;++i) max_abs_O = std::max(max_abs_O, (double)std::fabs(O[i]-O_ref[i]));
        for (int i=0;i<N;++i)   max_abs_L = std::max(max_abs_L, (double)std::fabs(L[i]-L_ref[i]));
        printf("\n[Check vs CPU] N=%d d=%d  max|O-O_ref|=%.3e  max|L-L_ref|=%.3e\n",
               N,d, max_abs_O, max_abs_L);
    }

    return 0;
}
