#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

typedef struct {
    const float *A;
    const float *B;
    float *C;
    int M, K, N;
    int row_begin;
    int row_end;
} Task;

void* worker(void *arg) {
    Task *t = (Task*)arg;
    const float *A = t->A, *B = t->B;
    float *C = t->C;
    int K = t->K, N = t->N;

    for (int i = t->row_begin; i < t->row_end; ++i) {
        for (int j = 0; j < N; ++j){
            float acc = 0.0f;
            for (int p = 0; p < K; ++p){
                acc += A[i*K + p] * B[p*N + j];
            }
            C[i*N + j] = acc;
        }
    }
    return NULL;
}

void matmul_pthreads(const float *A, const float *B, float *C, int M, int K, int N, int nthreads) {
    if (nthreads > M) nthreads = M;
    pthread_t *threads = malloc(sizeof(pthread_t) * nthreads);
    Task *tasks = malloc(sizeof(Task) * nthreads);

    int base = M / nthreads, rem = M % nthreads;
    int row = 0;
    for (int t=0; t < nthreads; ++t) {
        int rows_for_thread = base + (t<rem ? 1:0);
        tasks[t] = (Task){A, B, C, M, K, N, row, row + rows_for_thread};
        row += rows_for_thread;
        pthread_create(&threads[t], NULL, worker, &tasks[t]);
    }
    for (int t = 0; t < nthreads; ++t){
        pthread_join(threads[t], NULL);
    }
    free(threads);
    free(tasks);
}

double wall_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9; 
}

int main() {
    int M = 1000, K = 1000, N = 1000;
    size_t sizeA = (size_t)M*K, sizeB = (size_t)K*N, sizeC = (size_t)M*N;
    float *A = malloc(sizeof(float) * sizeA);
    float *B = malloc(sizeof(float) * sizeB);
    float *C = malloc(sizeof(float) * sizeC);

    for (size_t i=0;i<sizeA;++i) A[i] = (float)(rand()%100 + 1);
    for (size_t i=0;i<sizeB;++i) B[i] = (float)(rand()%100 + 1);

    int thread_counts[] = {1,2,4,8,16,32};
    double baseline = 0.0;

    for (int t=0; t < 6; ++t) {
        int nthreads = thread_counts[t];
        double elapsed_sum = 0.0;
        int repeats = 3; 
        for (int rep=0; rep<repeats; ++rep) {
            double t0 = wall_time();
            matmul_pthreads(A,B,C,M,K,N,nthreads);
            double t1 = wall_time();
            elapsed_sum += (t1 - t0);
        }
        double elapsed = elapsed_sum / repeats;
        if (nthreads == 1) baseline = elapsed;
        double speedup = baseline / elapsed;
        printf("Threads=%3d  Time=%.6f sec  Speedup=%.2f\n", nthreads, elapsed, speedup);
    }

    printf("Check values: C[0]=%.8e, C[last]=%.8e\n", C[0], C[M*N-1]);

    free(A); free(B); free(C);
    return 0;

}