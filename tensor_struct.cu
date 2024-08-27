#include<stdio.h>
#include<stdlib.h>
#include<cuda-runtime.h>


#include<matrix_struct.h>

typedef struct Tensor{
    float *T;
    int *dim;
    int *stride;
    char device_type;
}Tensor;

Tensor *makeTensor(int *dim, int *stride,int num_dim, int device_type){
    Tensor* ten = (Tensor*)malloc(sizeof(Tensor));
    if(!device_type){
        ten->T = (float*)malloc(row * col *sizeof(float));
    }else{
        cudaSetDevice(device_type-1);
    }
}


void tensor_matmul_(float *dA, float *dB, float *dC, float *dBias, int M, int N, int K) {
    // dim3 dimGrid((N + tile_SIZE - 1) / tile_SIZE, (M + tile_SIZE - 1) / tile_SIZE, dim); //dim은 4x3x32x32를 matmul하는 경우 12가 들어가게 된다.
    dim3 dimBlock(tile_SIZE, tile_SIZE);
    tiledMM<<<dimGrid, dimBlock>>>(dA, dB, dC, dBias, M, N, K);
}