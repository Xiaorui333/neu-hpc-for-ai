#pragma once

#include <stdio.h>

#ifdef NDEBUG
    #define assertc(ignore) ((void)0)
#else
    #ifdef __CUDA_ARCH__
        // Device-side assertion (GPU)
        #define assertc(expression) \
        do { \
        if (!(expression)) { \
        printf("Device assertion failed: (%s), function %s, file %s, line %d.\n", \
        #expression, __func__, __FILE__, __LINE__); \
        __trap(); \
        } \
        } while(0)
    #else
        // Host-side assertion (CPU)
        #define assertc(expression) \
        do { \
        if (!(expression)) { \
        fprintf(stderr, "Host assertion failed: (%s), function %s, file %s, line %d.\n", \
        #expression, __func__, __FILE__, __LINE__); \
        fflush(stderr); \
        *(volatile int*)0 = 0; \
        } \
        } while(0)
    #endif
#endif
