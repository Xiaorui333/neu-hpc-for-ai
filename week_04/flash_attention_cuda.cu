#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <math_constants.h>  

#define CHECK_CUDA(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)

// Toggle logits scaling (1/sqrt(d)).
#ifndef USE_SCALE
#define USE_SCALE 1
#endif

// ===== Kernel =================================================================

static inline __device__ float neg_inf() { return -CUDART_INF_F; }

// Forward kernel. One thread per row inside the Qi tile.
__global__ void flash_attn_fwd_kernel(
    const float* __restrict__ Q,   // [N, d]
    const float* __restrict__ K,   // [N, d]
    const float* __restrict__ V,   // [N, d]
    float* __restrict__ O,         // [N, d]
    float* __restrict__ l,         // [N]  running normalizer
    float* __restrict__ m,         // [N]  running row-max
    int N, int d,
    int Br, int Bc, int Tc,
    float scale                    // 1/sqrt(d) if enabled
){
    extern __shared__ float smem[];
    // Shared memory layout:
    // [ Qi (Br*d) | Oi (Br*d) | li (Br) | mi (Br) | Kj (Bc*d) | Vj (Bc*d) | tmax (Br) | tsum (Br) ]
    float* Qi   = smem;
    float* Oi   = Qi + Br*d;
    float* li   = Oi + Br*d;
    float* mi   = li + Br;
    float* Kj   = mi + Br;
    float* Vj   = Kj + Bc*d;
    float* tmax = Vj + Bc*d;
    float* tsum = tmax + Br;

    const int row_start = blockIdx.x * Br;
    const int r_local   = threadIdx.x;      
    const int row       = row_start + r_local;

    const int Br_eff = min(Br, N - row_start);     // effective rows in this tile
    if (r_local >= Br_eff || row >= N) return;

    // Load Qi[row,:] and init Oi[row,:]=0 (Alg. 1 line 8)
    for (int c = 0; c < d; ++c) {
        Qi[r_local*d + c] = Q[row*d + c];
        Oi[r_local*d + c] = 0.0f;
    }
    __syncthreads();

    // Load initial running stats (Alg. 1 line 8)
    mi[r_local] = m[row];
    li[r_local] = l[row];
    __syncthreads();

    // Iterate all KV tiles
    for (int j = 0; j < Tc; ++j) {
        const int col_start = j * Bc;
        const int Bc_eff = min(Bc, N - col_start);

        // Load Kj, Vj from HBM to shared (cooperative among Br threads)
        for (int t = r_local; t < Bc_eff*d; t += Br_eff) {
            int r = t / d, c = t % d;
            Kj[r*d + c] = K[(col_start + r)*d + c];
            Vj[r*d + c] = V[(col_start + r)*d + c];
        }
        __syncthreads();

        // ---- Pass 1: row-wise max over current KV tile (m̃_ij) ----
        float my_max = neg_inf();
        for (int c = 0; c < Bc_eff; ++c) {
            const float* qptr = &Qi[r_local*d];
            const float* kptr = &Kj[c*d];
            float dot = 0.f;
            #pragma unroll 1
            for (int t = 0; t < d; ++t) dot += qptr[t] * kptr[t];
#if USE_SCALE
            float s = dot * scale;
#else
            float s = dot;
#endif
            my_max = fmaxf(my_max, s);
        }
        tmax[r_local] = my_max;
        __syncthreads();

        // ---- Pass 2: compute ℓ̃_ij = sum_c exp(s - m̃_ij) ----
        float l_t = 0.f;
        for (int c = 0; c < Bc_eff; ++c) {
            const float* qptr = &Qi[r_local*d];
            const float* kptr = &Kj[c*d];
            float dot = 0.f;
            #pragma unroll 1
            for (int t = 0; t < d; ++t) dot += qptr[t] * kptr[t];
#if USE_SCALE
            float s = dot * scale;
#else
            float s = dot;
#endif
            l_t += __expf(s - tmax[r_local]);       
        }
        tsum[r_local] = l_t;
        __syncthreads();

        // ---- Online merge of (m,l) with (m̃,ℓ̃) (Alg. 1 line 11) ----
        float mi_old = mi[r_local];
        float li_old = li[r_local];
        float m_t    = tmax[r_local];
        float m_new  = fmaxf(mi_old, m_t);
        float l_new  = __expf(mi_old - m_new) * li_old + __expf(m_t - m_new) * l_t;

        // Row-wise coefficients (Alg. 1 line 12), applied ONCE per tile
        float alpha = (l_new > 0.f) ? (__expf(mi_old - m_new) * (li_old / l_new)) : 0.f;
        float beta  = (l_new > 0.f) ? ( __expf(m_t    - m_new) /  l_new )         : 0.f;

        // ---- Build acc[t] = Σ_c exp(s - m_t) * Vj[c,t]  ----
        for (int t = 0; t < d; ++t) {
            float acc = 0.f;
            for (int c = 0; c < Bc_eff; ++c) {
                const float* qptr = &Qi[r_local*d];
                const float* kptr = &Kj[c*d];
                float dot = 0.f;
                #pragma unroll 1
                for (int tt = 0; tt < d; ++tt) dot += qptr[tt] * kptr[tt];
#if USE_SCALE
                float s = dot * scale;
#else
                float s = dot;
#endif
                float w_un = __expf(s - m_t);       
                acc += w_un * Vj[c*d + t];
            }
            // ---- O_i = alpha * O_i + beta * acc ----
            float old = Oi[r_local*d + t];
            Oi[r_local*d + t] = alpha * old + beta * acc;
        }

        // ---- Commit new running stats (Alg. 1 line 13) ----
        mi[r_local] = (l_new > 0.f) ? m_new : mi_old;  
        li[r_local] = l_new;
        __syncthreads();
    }

    // Write back O_i, l_i, m_i
    for (int c = 0; c < d; ++c)
        O[row*d + c] = Oi[r_local*d + c];
    l[row] = li[r_local];
    m[row] = mi[r_local];
}

// ===== Host launcher ===========================================================

void flash_attn_forward_cuda(
    const std::vector<float>& Qh,
    const std::vector<float>& Kh,
    const std::vector<float>& Vh,
    int N, int d)
{
    size_t bytes = (size_t)N * d * sizeof(float);
    float *Q,*K,*V,*O,*l,*m;
    CHECK_CUDA(cudaMalloc(&Q, bytes));
    CHECK_CUDA(cudaMalloc(&K, bytes));
    CHECK_CUDA(cudaMalloc(&V, bytes));
    CHECK_CUDA(cudaMalloc(&O, bytes));
    CHECK_CUDA(cudaMalloc(&l, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&m, N*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(Q, Qh.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(K, Kh.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(V, Vh.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(O, 0, bytes));

    // Initialize running stats: l=0, m=-inf
    {
        std::vector<float> lh(N, 0.f), mh(N, -INFINITY);
        CHECK_CUDA(cudaMemcpy(l, lh.data(), N*sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(m, mh.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    }

    // Get per-block shared memory budget (in floats)
    int device; CHECK_CUDA(cudaGetDevice(&device));
    int smemPerBlockBytes = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&smemPerBlockBytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    if (smemPerBlockBytes == 0)
        CHECK_CUDA(cudaDeviceGetAttribute(&smemPerBlockBytes, cudaDevAttrMaxSharedMemoryPerBlock, device));
    size_t M_f = (size_t)smemPerBlockBytes / sizeof(float);

    // floats_per_block = (2*Br + 2*Bc)*d + (4*Br)
    int Bc = (int)fmin((double)N, floor((double)M_f / (4.0 * d)));
    int Br = (int)fmin((double)d, floor((double)M_f / (4.0 * d)));
    if (Bc <= 0) Bc = 1;
    if (Br <= 0) Br = 1;

    auto fits = [&](int BrT, int BcT)->bool{
        size_t need = (size_t)((2*BrT + 2*BcT)*d + (4*BrT));
        return need <= M_f;
    };
    while (!fits(Br, Bc)) {
        if (Bc > 1) --Bc;
        if (!fits(Br, Bc) && Br > 1) --Br;
        if (Bc==1 && Br==1) break;
    }

    int Tr = (N + Br - 1) / Br;
    int Tc = (N + Bc - 1) / Bc;

    dim3 grid(Tr, 1, 1);
    dim3 block(Br, 1, 1);

    size_t shmem_bytes = sizeof(float) * ((2*Br + 2*Bc)*d + (4*Br));
    CHECK_CUDA(cudaFuncSetAttribute(flash_attn_fwd_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)shmem_bytes));

    float scale = 1.0f / sqrtf((float)d);
#if !USE_SCALE
    scale = 1.0f;
#endif

    flash_attn_fwd_kernel<<<grid, block, shmem_bytes>>>(Q,K,V,O,l,m, N,d, Br,Bc,Tc, scale);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Simple sanity print: print both rows (if N>=2)
    std::vector<float> Oh(N*d);
    CHECK_CUDA(cudaMemcpy(Oh.data(), O, bytes, cudaMemcpyDeviceToHost));
    for (int i=0;i<N;i++) {
        printf("O[%d]: ", i);
        for (int t=0; t<d; ++t) printf("%.5f ", Oh[i*d + t]);
        printf("\n");
    }

    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O); cudaFree(l); cudaFree(m);
}

// ===== Demo ===================================================================

int main(){
    int N=2, d=4;
    std::vector<float> Q = {1,0,1,0,  0,1,0,1};
    std::vector<float> K = {1,0,1,0,  0,1,0,1};
    std::vector<float> V = {10,20,30,40,  50,60,70,80};

    printf("FlashAttention (NO MASK), USE_SCALE=%d (1/sqrt(d)=%.6f)\n",
           USE_SCALE, 1.0f/sqrtf((float)d));

    flash_attn_forward_cuda(Q,K,V, N,d);
    return 0;
}
