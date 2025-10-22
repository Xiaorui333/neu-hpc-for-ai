#pragma once

#include <string.h>
#include "assertc.h"
#include "min.h"
#include <stdio.h>

#include "mat_util.h"


typedef unsigned int uint;

typedef struct {
    uint x;
    uint y;
    uint z;
} dim3;

dim3 dim(uint x, uint y) {
    dim3 d = {x, y, 1};
    return d;
}

dim3 idx(uint x, uint y) {
    dim3 d = {x, y, 0};
    return d;
}

typedef struct {
    float* data;   // underlying matrix buffer
    dim3  idx;     // start coordinate in the underlying matrix
    dim3  size;    // extents of THIS tile (x = rows, y = cols)
    unsigned int ld; // leading dimension (row stride) of underlying matrix
} tile;

tile tile_alloc(unsigned int rows, unsigned int cols) {
    tile t;
    t.data = (float*)malloc(rows * cols * sizeof(float));
    t.idx  = (dim3){0,0,0};
    t.size = (dim3){rows, cols, 1};
    t.ld   = cols;                 // IMPORTANT: stride = full matrix width
    return t;
}

unsigned int tile_data_idx(tile t, unsigned int i, unsigned int j) {
    assertc(i < t.size.x);
    assertc(j < t.size.y);
    unsigned int x = t.idx.x + i;
    unsigned int y = t.idx.y + j;
    return x * t.ld + y;           // use ld, not size.y
}

float tile_at(tile t, unsigned int i, unsigned int j) {
    return t.data[tile_data_idx(t, i, j)];
}

void tile_set(tile t, unsigned int i, unsigned int j, float v) {
    t.data[tile_data_idx(t, i, j)] = v;
}

void tilecpy(tile dst, tile src) {
    assertc(dst.size.x == src.size.x);
    assertc(dst.size.y == src.size.y);
    for (unsigned int i = 0; i < src.size.x; ++i)
        for (unsigned int j = 0; j < src.size.y; ++j)
            tile_set(dst, i, j, tile_at(src, i, j));
}

void tile_print(tile t) {
    for (unsigned int i = 0; i < t.size.x; ++i) {
        for (unsigned int j = 0; j < t.size.y; ++j)
            printf("%.02f\t", tile_at(t, i, j));
        printf("\n");
    }
}

unsigned int umin(unsigned int a, unsigned int b) {
    return a < b ? a : b;
}

tile sub_tile(tile a, dim3 subIdx, dim3 subDim) {
    tile t = a; // shares data and ld

    // start of the sub-tile in the underlying matrix
    unsigned int sx = a.idx.x + subIdx.x * subDim.x;
    unsigned int sy = a.idx.y + subIdx.y * subDim.y;

    t.idx.x = sx;
    t.idx.y = sy;

    // remaining room inside parent tile starting at (sx, sy)
    unsigned int rx = (a.idx.x + a.size.x > sx) ? (a.idx.x + a.size.x - sx) : 0;
    unsigned int ry = (a.idx.y + a.size.y > sy) ? (a.idx.y + a.size.y - sy) : 0;

    // clip extents
    t.size.x = umin(subDim.x, rx);
    t.size.y = umin(subDim.y, ry);

    // ld unchanged
    return t;
}
