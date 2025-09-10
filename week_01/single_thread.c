#include <assert.h>
#include <stdio.h>

// C = A(MxK) * B(KxN) ；A[i*K+p]、B[p*N+j]、C[i*N+j]
void matmul(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int p = 0; p < K; ++p) {
                acc += A[i*K + p] * B[p*N + j];
            }
            C[i*N + j] = acc;
        }
    }
}

int main(void) {
    // A = 1x1, B = 1x1
    { float A[1]={2}, B[1]={3}, C[1]={0};
      matmul(A,B,C,1,1,1);
      assert(C[0]==6.0f);
    }
    // A = 1x1, B = 1x5
    { float A[1]={1}, B[5]={1,2,3,4,5}, C[5]={0};
      matmul(A,B,C,1,1,5);
      for (int j=0;j<5;++j) assert(C[j]==B[j]);
    }
    // A = 2x1, B = 1x3
    { float A[2]={1,2}, B[3]={1,2,3}, C[6]={0};
      float E[6]={1,2,3, 2,4,6};
      matmul(A,B,C,2,1,3);
      for (int i=0;i<6;++i) assert(C[i]==E[i]);
    }
    // A = 2x2, B = 2x2
    { float A[4]={1,2,3,4}, B[4]={5,6,7,8}, C[4]={0};
      float E[4]={19,22,43,50};
      matmul(A,B,C,2,2,2);
      for (int i=0;i<4;++i) assert(C[i]==E[i]);
    }

    puts("All minimal tests passed.");
    return 0;
}
