#pragma once
#include <stdio.h>

size_t  div_ceil(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}
