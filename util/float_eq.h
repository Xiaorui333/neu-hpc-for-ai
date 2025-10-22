#pragma once

#include <stdbool.h>
#include "color.h"

int float_equal(float a, float b) {
    float x = a - b;
    x = (x < 0.0f) ? -x : x;
    return x < 1e-3f;
}

void check_float_array_eq(float* actual, float* expected, size_t N) {
    for (int i = 0; i < N; i++) {
        if (!float_equal(actual[i], expected[i])) {
            red("assert_float_array_eq failed:\n");
            red("actual[%d] = %f, expected[%d] = %f\n", i, actual[i], i, expected[i]);
            red("FAILED\n");
            return;
        }
    }
    green("SUCCESS\n");
}
