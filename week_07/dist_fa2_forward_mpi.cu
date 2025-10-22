#include <mpi.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

#ifndef USE_SCALE
#define USE_SCALE 1
#endif

#define CHECK_CUDA(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 1);} } while(0)

#define CHECK_MPI(x) do { int e=(x); if(e!=MPI_SUCCESS){ \
  fprintf(stderr,"MPI %s:%d: MPI error %d\n", __FILE__, __LINE__, e); MPI_Abort(MPI_COMM_WORLD, 1);} } while(0)

__device__ __forceinline__ float neg_inf() { return -CUDART_INF_F; }

__host__ __device__ __forceinline__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}


// Local FA-2 kernel
__global__ void fa2_local_kernel(
    const float* __restrict__ Qi,   // [Br,d]
    const float* __restrict__ Ksh,  // [Nsh,d]
    const float* __restrict__ Vsh,  // [Nsh,d]
    float* __restrict__ Otilde,     // [Br,d]  (unnormalized)
    float* __restrict__ m_loc,      // [Br]
    float* __restrict__ ell_loc,    // [Br]
    int Br, int d, int Nsh, int Bc, float scale)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x; 
    if (r >= Br) return;

    const float* qrow = Qi + (size_t)r*d;
    // Init FA-2 state (for this device)
    float m = neg_inf();  
    float l = 0.f;        
    // Otilde accumulates unnormalized PV with proper FA-2 rebasing
    for (int t=0; t<d; ++t) Otilde[r*(size_t)d + t] = 0.f;

    int Tc_loc = div_up(Nsh, Bc);
    for (int j=0; j<Tc_loc; ++j){
        int col0   = j * Bc;
        int Bc_eff = max(0, min(Bc, Nsh - col0));

        // Compute per-tile rowwise max m_t
        float m_t = neg_inf();
        for (int c=0; c<Bc_eff; ++c){
            const float* kptr = Ksh + (size_t)(col0 + c)*d;
            float dot=0.f;
            #pragma unroll 1
            for (int u=0; u<d; ++u) dot += qrow[u]*kptr[u];
#if USE_SCALE
            dot *= scale;
#endif
            m_t = fmaxf(m_t, dot);
        }

        // Compute l_t = sum_c exp(s - m_t)
        float l_t = 0.f;
        for (int c=0; c<Bc_eff; ++c){
            const float* kptr = Ksh + (size_t)(col0 + c)*d;
            float dot=0.f;
            #pragma unroll 1
            for (int u=0; u<d; ++u) dot += qrow[u]*kptr[u];
#if USE_SCALE
            dot *= scale;
#endif
            l_t += __expf(dot - m_t);
        }

        // Online merge (FA-2) softmax
        float m_new = fmaxf(m, m_t);
        float alpha = __expf(m - m_new);   // scale previous state
        float beta  = __expf(m_t - m_new); // scale this tile
        float l_new = alpha * l + beta * l_t;

        // Rescale previous Otilde 
        if (l > 0.f) {
            for (int t=0; t<d; ++t)
                Otilde[r*(size_t)d + t] *= alpha;
        }

        // Add current tile
        for (int t=0; t<d; ++t){
            float acc = 0.f;
            for (int c=0; c<Bc_eff; ++c){
                const float* kptr = Ksh + (size_t)(col0 + c)*d;
                const float* vptr = Vsh + (size_t)(col0 + c)*d;
                float dot=0.f;
                #pragma unroll 1
                for (int u=0; u<d; ++u) dot += qrow[u]*kptr[u];
#if USE_SCALE
                dot *= scale;
#endif
                float w_un = __expf(dot - m_t); 
                acc += w_un * vptr[t];
            }
            Otilde[r*(size_t)d + t] += beta * acc;
        }

        m = m_new; l = l_new;
    }

    m_loc[r]   = m;
    ell_loc[r] = l; // still in local base
}

// Rebase
__global__ void rebase_locals_kernel(
    float* __restrict__ Otilde,   // [Br,d]
    float* __restrict__ ell_loc,  // [Br]
    const float* __restrict__ m_loc,
    const float* __restrict__ m_glob,
    int Br, int d)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= Br) return;

    float shift = __expf(m_loc[r] - m_glob[r]);
    for (int t=0; t<d; ++t)
        Otilde[r*(size_t)d + t] *= shift;
    ell_loc[r] *= shift;
}

/* Finalize:
   O = Otilde_glob / ell_glob
   L = m_glob + log(max(ell_glob, 1e-38))
*/
__global__ void finalize_kernel(
    const float* __restrict__ Otilde_glob, // [Br,d]
    const float* __restrict__ ell_glob,    // [Br]
    const float* __restrict__ m_glob,      // [Br]
    float* __restrict__ O_out,             // [Br,d]
    float* __restrict__ L_out,             // [Br]
    int Br, int d)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= Br) return;
    float l = ell_glob[r];
    float inv = (l > 0.f) ? 1.0f/l : 0.0f;
    for (int t=0; t<d; ++t)
        O_out[r*(size_t)d + t] = Otilde_glob[r*(size_t)d + t] * inv;
    L_out[r] = m_glob[r] + logf(fmaxf(l, 1e-38f));
}

/* Utility: host-side data init (rank 0) */
static void make_toy_inputs(int N, int d,
                            std::vector<float>& Q,
                            std::vector<float>& K,
                            std::vector<float>& V)
{
    Q.resize((size_t)N*d);
    K.resize((size_t)N*d);
    V.resize((size_t)N*d);
    
    for (int i=0;i<N;i++){
        for (int t=0;t<d;t++){
            Q[(size_t)i*d+t] = (float)((i+t)%7 - 3) * 0.1f;
            K[(size_t)i*d+t] = (float)((i*3+t)%5 - 2) * 0.2f;
            V[(size_t)i*d+t] = (float)((i*5+t)%11 - 5) * 1.0f;
        }
    }
}

int main(int argc, char** argv){
    CHECK_MPI(MPI_Init(&argc, &argv));
    int rank=0, world=1;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &world));

    // ------------------ Parse args ------------------
    int N = 1024, d = 128, Br = 128, Bc = 128;
    for (int i=1;i<argc;i++){
        if (!strcmp(argv[i],"--N") && i+1<argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--d") && i+1<argc) d = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--Br") && i+1<argc) Br = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--Bc") && i+1<argc) Bc = atoi(argv[++i]);
    }

    if (N < 1 || world < 1 || world > 8){
        if (rank==0) fprintf(stderr,"N>=1 and 1<=world<=8 required.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);   
    }

    // Map rank -> GPU
    CHECK_CUDA(cudaSetDevice(rank));

    // ------------------ Create / scatter inputs ------------------
    std::vector<float> hQ, hK, hV;
    if (rank == 0)
        make_toy_inputs(N, d, hQ, hK, hV);

    // Shard K,V rows across ranks
    int rows_per_rank = div_up(N, world);
    int row_beg = rank * rows_per_rank;
    int row_end = std::min(N, row_beg + rows_per_rank);
    int Nsh = std::max(0, row_end - row_beg);

    // Scatter K,V shards via host buffers
    std::vector<int> counts(world), displs(world);
    if (rank == 0){
        for (int r=0; r<world; r++){
            int rb = r * rows_per_rank;
            int re = std::min(N, rb + rows_per_rank);
            counts[r] = (re - rb) * d;
            displs[r] = rb * d;
        }
    }

    std::vector<float> hKsh((size_t)Nsh*d), hVsh((size_t)Nsh*d);

    CHECK_MPI(MPI_Scatterv(
        rank==0 ? hK.data() : nullptr,
        counts.data(), displs.data(), MPI_FLOAT,
        hKsh.data(), (int)hKsh.size(), MPI_FLOAT,
        0, MPI_COMM_WORLD));
    

    CHECK_MPI(MPI_Scatterv(
    rank==0 ? hV.data() : nullptr,
    counts.data(), displs.data(), MPI_FLOAT,
    hVsh.data(), (int)hVsh.size(), MPI_FLOAT,
    0, MPI_COMM_WORLD));

    // ------------------ Allocate device buffers ------------------
    float *dKsh=nullptr, *dVsh=nullptr;
    CHECK_CUDA(cudaMalloc(&dKsh, (size_t)Nsh*d*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dVsh, (size_t)Nsh*d*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dKsh, hKsh.data(), (size_t)Nsh*d*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dVsh, hVsh.data(), (size_t)Nsh*d*sizeof(float), cudaMemcpyHostToDevice));

    float scale = USE_SCALE ? (1.0f / sqrtf((float)d)) : 1.0f;

    // Per-tile device buffers (same on all ranks)
    float *dQi=nullptr, *dOtilde_loc=nullptr, *dm_loc=nullptr, *dl_loc=nullptr;
    float *dOtilde_glob=nullptr, *dm_glob=nullptr, *dl_glob=nullptr;
    float *dO_tile=nullptr, *dL_tile=nullptr;
    CHECK_CUDA(cudaMalloc(&dQi,            (size_t)Br*d*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOtilde_loc,    (size_t)Br*d*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dm_loc,         (size_t)Br*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dl_loc,         (size_t)Br*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOtilde_glob,   (size_t)Br*d*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dm_glob,        (size_t)Br*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dl_glob,        (size_t)Br*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dO_tile,        (size_t)Br*d*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dL_tile,        (size_t)Br*sizeof(float)));

    // Final outputs on rank 0
    float *dO_all=nullptr, *dL_all=nullptr;
    if (rank==0){
        CHECK_CUDA(cudaMalloc(&dO_all, (size_t)N*d*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dL_all, (size_t)N*sizeof(float)));
    }

    // ------------------ Main loop over Q tiles ------------------
    int Tr = div_up(N, Br);
    dim3 rowGrid(div_up(Br, 256));

    for (int it=0; it<Tr; ++it){
        int row0   = it * Br;
        int Br_eff = std::min(Br, N - row0);
        if (Br_eff <= 0) break;

        // Rank 0: copy Qi from full Q to device; others: dummy
        if (rank == 0){
            CHECK_CUDA(cudaMemcpy(dQi, hQ.data() + (size_t)row0*d, (size_t)Br_eff*d*sizeof(float),cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // MPI broadcast of device buffer dQi
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_MPI(MPI_Bcast(dQi, Br_eff*d, MPI_FLOAT, 0, MPI_COMM_WORLD));
        CHECK_CUDA(cudaDeviceSynchronize());

        // Local FA-2 on this shard
        {
            dim3 grid(div_up(Br_eff,256)),block(256);
            fa2_local_kernel<<<grid, block>>>(
                dQi, dKsh, dVsh, dOtilde_loc, dm_loc, dl_loc,
                Br_eff, d, Nsh, Bc, scale);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // Global rowwise max of m: dm_glob = max(dm_loc)
        CHECK_MPI(MPI_Allreduce(dm_loc, dm_glob, Br_eff, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD));

        // Rebase locals to global m (in-place)
        {
            dim3 grid(div_up(Br_eff,256)), block(256);
            rebase_locals_kernel<<<grid, block>>>(
                dOtilde_loc, dl_loc, dm_loc, dm_glob, Br_eff, d
            );
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // Sum across ranks: ell_glob, Otilde_glob
        CHECK_MPI(MPI_Allreduce(dl_loc, dl_glob, Br_eff, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
        CHECK_MPI(MPI_Allreduce(dOtilde_loc, dOtilde_glob, Br_eff*d, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));

        // Finalize this tile to O and L
        {
            dim3 grid(div_up(Br_eff, 256)), block(256);
            finalize_kernel<<<grid, block>>>(
                dOtilde_glob, dl_glob, dm_glob, dO_tile, dL_tile, Br_eff, d);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // Rank 0 gathers tile outputs into full O,L
        if (rank == 0){
            CHECK_CUDA(cudaMemcpy(dO_all + (size_t)row0*d, dO_tile,
                                  (size_t)Br_eff*d*sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(dL_all + row0, dL_tile,
                                  (size_t)Br_eff*sizeof(float), cudaMemcpyDeviceToDevice));
        }
    }

    // ------------------ Print a small slice (rank 0) ------------------
    if (rank == 0){
        std::vector<float> hO(N*(size_t)d), hL(N);
        CHECK_CUDA(cudaMemcpy(hO.data(), dO_all, (size_t)N*d*sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(hL.data(), dL_all, (size_t)N*sizeof(float),   cudaMemcpyDeviceToHost));
        int show = std::min(N, 2);
        printf("Distributed FA-2 forward (MPI), N=%d d=%d Br=%d Bc=%d (world=%d) USE_SCALE=%d\n",
               N,d,Br,Bc,world, USE_SCALE);
        for (int i=0;i<show;i++){
            printf("O[%d,0:8): ", i);
            for (int t=0;t<std::min(d,8); ++t) printf("%.5f ", hO[(size_t)i*d + t]);
            printf(" | L=%.6f\n", hL[i]);
        }
    }

    // ------------------ Cleanup ------------------
    if (dKsh) cudaFree(dKsh); if (dVsh) cudaFree(dVsh);
    cudaFree(dQi); cudaFree(dOtilde_loc); cudaFree(dm_loc); cudaFree(dl_loc);
    cudaFree(dOtilde_glob); cudaFree(dm_glob); cudaFree(dl_glob);
    cudaFree(dO_tile); cudaFree(dL_tile);
    if (rank==0){ cudaFree(dO_all); cudaFree(dL_all); }

    MPI_Finalize();
    return 0;
}




