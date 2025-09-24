// online_softmax.c
// Algorithm 3: Safe softmax with online normalizer calculation (sequential)

#include <math.h>
#include <stddef.h>

#define SOFTMAX_T    float
#define SOFTMAX_EXP  expf     
#define SOFTMAX_FMAX fmaxf  


void softmax_online(const SOFTMAX_T *x, size_t n, SOFTMAX_T *y) {
    if (n == 0) return;

    SOFTMAX_T m = -INFINITY; 
    SOFTMAX_T d = 0;         

    for (size_t j = 0; j < n; ++j) {
        SOFTMAX_T mj = SOFTMAX_FMAX(m, x[j]);                         
        d = d * SOFTMAX_EXP(m - mj) + SOFTMAX_EXP(x[j] - mj);       
        m = mj;
    }

    for (size_t i = 0; i < n; ++i) {
        y[i] = SOFTMAX_EXP(x[i] - m) / d;                         
    }
}

#ifdef TEST_ONLINE_SOFTMAX
#include <stdio.h>

static SOFTMAX_T sum_arr(const SOFTMAX_T *a, size_t n) {
    SOFTMAX_T s = 0;
    for (size_t i = 0; i < n; i++) s += a[i];
    return s; 
}

int main(void) {
    SOFTMAX_T x[] = {1000.f, 1002.f, 999.f, -5.f, 0.f};
    size_t n = sizeof(x) / sizeof(x[0]);
    SOFTMAX_T y[5];

    softmax_online(x, n, y);

    printf("Softmax:\n");
    for (size_t i = 0; i < n; ++i) {
        printf("y[%zu] = %.9f\n", i, y[i]);
    }
    printf("sum = %.9f\n", sum_arr(y, n));
    return 0;
}
#endif
