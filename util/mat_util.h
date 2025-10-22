#pragma once

#include <stdlib.h>
#include "assertc.h"

size_t at(size_t r, size_t c, size_t rows, size_t cols) {
    assertc(r < rows);
    assertc(c < cols);
    return r * cols + c;
}

void mat_print(char* name, float *A, size_t M, size_t N) {
    printf("%s:\n", name);
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            printf("%f ", A[i * N + j]);
        }
        printf("\n");
    }
}
