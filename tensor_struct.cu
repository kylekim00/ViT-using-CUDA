#include<stdio.h>
#include<stdlib.h>
#include<matrix_struct.h>

typedef struct Tensor{
    Matrix **matrices;
    int *dim;
    int *stride;
    
}

void tensormatmul_(float *dA, float *dB, float *dC, float *dBias, int M, int N, int K) {
    // dim3 dimGrid((N + tile_SIZE - 1) / tile_SIZE, (M + tile_SIZE - 1) / tile_SIZE, dim); //dim은 4x3x32x32를 matmul하는 경우 12가 들어가게 된다.
    dim3 dimBlock(tile_SIZE, tile_SIZE);
    tiledMM<<<dimGrid, dimBlock>>>(dA, dB, dC, dBias, M, N, K);
}