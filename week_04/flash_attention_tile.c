// flash_attention_tile.c
// Minimal CPU-side, tile-by-tile FlashAttention forward


#include <math.h>
#include <string.h>

#include "util/assertc.h"
#include "util/trace.h"
#include "util/float_eq.h"
#include "util/mat_util.h"
#include "util/div_ceil.h"
#include "util/min.h"
#include "util/tile.h"


typedef unsigned int uint;

/* Compute one (Qi, Oi) block by iterating over all (Kj, Vj) tiles.
 * This mirrors Algorithm 1 lines 9–13 with online softmax state (m_i, l_i) per row. */
static void flash_attention_block(tile Q, tile K, tile V, tile O,
                                  uint N, uint d,
                                  uint i, uint Br, uint Bc, uint Tc)
{
    // ---- Views for this query tile 
    tile Qi = sub_tile(Q, idx(i, 0), dim(Br, d));

    // ---- Allocate shared/on-chip buffers for this query tile (use effective sizes)
    const uint Br_eff = Qi.size.x;   // rows of this Q tile
    const uint d_eff  = Qi.size.y;   // should equal d

    tile sQi   = tile_alloc(Br_eff, d_eff);
    tile sOi   = tile_alloc(Br_eff, d_eff);
    tile s_li  = tile_alloc(Br_eff, 1);
    tile s_mi  = tile_alloc(Br_eff, 1);

    tilecpy(sQi, Qi);

    // Initialize O_i, l_i, m_i for this query tile
    for (uint r = 0; r < Br_eff; ++r) {
        for (uint t = 0; t < d_eff; ++t) tile_set(sOi, r, t, 0.0f);
        tile_set(s_li, r, 0, 0.0f);
        tile_set(s_mi, r, 0, -INFINITY);
    }

    // Logits scaling 
    const float scale = 1.0f / sqrtf((float)d_eff);

    // ---- Iterate over all K/V tiles (columns)
    for (uint j = 0; j < Tc; ++j) {
        tile Kj = sub_tile(K, idx(j, 0), dim(Bc, d));
        tile Vj = sub_tile(V, idx(j, 0), dim(Bc, d));

        const uint Bc_eff = Kj.size.x;   // number of keys/values in this tile

        // Allocate per-KV-tile shared buffers with effective sizes
        tile sKj    = tile_alloc(Bc_eff, d_eff);
        tile sVj    = tile_alloc(Bc_eff, d_eff);
        tile sSij   = tile_alloc(Br_eff, Bc_eff);
        tile sPij   = tile_alloc(Br_eff, Bc_eff);
        tile s_mij  = tile_alloc(Br_eff, 1);
        tile s_lij  = tile_alloc(Br_eff, 1);
        tile s_mnew = tile_alloc(Br_eff, 1);
        tile s_lnew = tile_alloc(Br_eff, 1);

        tilecpy(sKj, Kj);
        tilecpy(sVj, Vj);

        /* -------- Algorithm 1, line 9: S_ij = Qi * Kj^T  -------- */
        for (uint r = 0; r < Br_eff; ++r) {
            for (uint c = 0; c < Bc_eff; ++c) {
                float dot = 0.0f;
                for (uint t = 0; t < d_eff; ++t) {
                    dot += tile_at(sQi, r, t) * tile_at(sKj, c, t);
                }
                tile_set(sSij, r, c, dot * scale);
            }
        }

        /* -- Algorithm 1, line 10: row-wise max, exp shift, and row-wise sum -- */
        for (uint r = 0; r < Br_eff; ++r) {
            // row max over current KV tile
            float row_max = -INFINITY;
            for (uint c = 0; c < Bc_eff; ++c) {
                float s_rc = tile_at(sSij, r, c);
                if (s_rc > row_max) row_max = s_rc;
            }
            tile_set(s_mij, r, 0, row_max);

            // exp shift + rowsum
            float row_sum = 0.0f;
            for (uint c = 0; c < Bc_eff; ++c) {
                float w = expf(tile_at(sSij, r, c) - row_max);
                tile_set(sPij, r, c, w);
                row_sum += w;
            }
            tile_set(s_lij, r, 0, row_sum);
        }

        /* ---- Algorithm 1, line 11: online-softmax merge (m_i, l_i) ---- */
        for (uint r = 0; r < Br_eff; ++r) {
            float mi  = tile_at(s_mi, r, 0);
            float li  = tile_at(s_li, r, 0);
            float mij = tile_at(s_mij, r, 0);
            float lij = tile_at(s_lij, r, 0);

            float mnew = fmaxf(mi, mij);
            float lnew = expf(mi  - mnew) * li + expf(mij - mnew) * lij;

            tile_set(s_mnew, r, 0, mnew);
            tile_set(s_lnew, r, 0, lnew);
        }

        /* -------- Algorithm 1, line 12: update O_i with current tile --------
         * O_i <- diag(ℓ_i^{new})^{-1} ( diag(ℓ_i) e^{m_i-m_i^{new}} O_i
         *           + e^{m̃_{ij}-m_i^{new}} P̃_{ij} V_j )
         * Implemented as:
         *   alpha = exp(mi-mnew) * (li / lnew)
         *   beta  = exp(mij-mnew) / lnew
         *   O_i   = alpha * O_i + beta * (P̃_ij @ V_j)
         */
        for (uint r = 0; r < Br_eff; ++r) {
            float mi   = tile_at(s_mi,  r, 0);
            float li   = tile_at(s_li,  r, 0);
            float mij  = tile_at(s_mij, r, 0);
            float lnew = tile_at(s_lnew, r, 0);
            float mnew = tile_at(s_mnew, r, 0);

            float alpha = (lnew > 0.0f) ? expf(mi  - mnew) * (li / lnew) : 0.0f;
            float beta  = (lnew > 0.0f) ? expf(mij - mnew) /  lnew       : 0.0f;

            for (uint t = 0; t < d_eff; ++t) {
                // acc = sum_c P̃_ij[r,c] * V_j[c,t] over current KV tile
                float acc = 0.0f;
                for (uint c = 0; c < Bc_eff; ++c) {
                    acc += tile_at(sPij, r, c) * tile_at(sVj, c, t);
                }
                float old = tile_at(sOi, r, t);
                tile_set(sOi, r, t, alpha * old + beta * acc);
            }
        }

        /* ---- Algorithm 1, line 13: commit online-softmax state ---- */
        for (uint r = 0; r < Br_eff; ++r) {
            tile_set(s_li, r, 0, tile_at(s_lnew, r, 0));
            tile_set(s_mi, r, 0, tile_at(s_mnew, r, 0));
        }

        // Free per-KV-tile buffers
        free(sKj.data);   free(sVj.data);
        free(sSij.data);  free(sPij.data);
        free(s_mij.data); free(s_lij.data);
        free(s_mnew.data); free(s_lnew.data);
    } 

    // ---- Write back O_i to global
    tile Oi = sub_tile(O, idx(i, 0), dim(Br, d));
    // Oi view may be smaller on the last tile; tilecpy handles the overlap
    tilecpy(Oi, sOi);

    // Free per-Qi-tile buffers
    free(sQi.data); free(sOi.data);
    free(s_li.data); free(s_mi.data);
}



void flash_attention(tile Q, tile K, tile V, tile O, uint N, uint d)
{
    uint Br = (N < 64) ? N : 64;   
    uint Bc = (N < 64) ? N : 64;   

    uint Tr = div_ceil(N, Br);
    uint Tc = div_ceil(N, Bc);

    for (uint i = 0; i < Tr; ++i) {
        flash_attention_block(Q, K, V, O, N, d, i, Br, Bc, Tc);
    }
}

int main(void)
{
    // Tiny sanity test (N=2, d=4)
    uint N = 2;
    uint d = 4;

    float Qdata[] = {
        1, 0, 1, 0,
        0, 1, 0, 1
    };
    float Kdata[] = {
        1, 0, 1, 0,
        0, 1, 0, 1
    };
    float Vdata[] = {
        10, 20, 30, 40,
        50, 60, 70, 80
    };

    float Oref[] = {
        20.75766f, 30.75766f, 40.75766f, 50.75766f,
        39.24234f, 49.24235f, 59.24235f, 69.24235f,
    };

    uint size = N * d * sizeof(float);

    tile Q = tile_alloc(N, d);
    tile K = tile_alloc(N, d);
    tile V = tile_alloc(N, d);
    tile O = tile_alloc(N, d);

    memcpy(Q.data, Qdata, size);
    memcpy(K.data, Kdata, size);
    memcpy(V.data, Vdata, size);

    flash_attention(Q, K, V, O, N, d);

    check_float_array_eq(O.data, Oref, N * d);

    free(Q.data); free(K.data); free(V.data); free(O.data);

    printf("OK\n");
    return 0;
}