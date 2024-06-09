#include<stdio.h>
#include<stdlib.h>
#include<matrix_struct.h>

// typedef struct Matrix{
//     float *M;
//     int row;
//     int col;
// }Matrix;
#define tile_SIZE 8
__global__ void flashAttention(float *dQ, float *dK, float *dV, int N, int K, int M){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float s_a[tile_SIZE][tile_SIZE];
    __shared__ float s_b[tile_SIZE][tile_SIZE];

    float tmp = 0.0f;

    for (int i = 0; i < (K + tile_SIZE - 1) / tile_SIZE; i++) {
        if (row < M && (i * tile_SIZE + threadIdx.x) < K)
            s_a[threadIdx.y][threadIdx.x] = dQ[row * K + (i * tile_SIZE + threadIdx.x)];
        else
            s_a[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (i * tile_SIZE + threadIdx.y) < K)
            s_b[threadIdx.y][threadIdx.x] = dK[(i * tile_SIZE + threadIdx.y) * N + col];
        else
            s_b[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int j = 0; j < tile_SIZE; j++) {
            tmp += s_a[threadIdx.y][j] * s_b[j][threadIdx.x];
        }

        __syncthreads();
    }
    if(threadIdx.x && threadIdx.y){//여기서 SOFTMax 처리를 해줄 것이다.

    }
    // if (row < M && col < N) {
    //     if (bias)
    //         C[row * N + col] = tmp + bias[col];
    //     else
    //         C[row * N + col] = tmp;
    // }
}

Matrix* convtoMatMul(Matrix *M, int input_size, int kernal_size, int stride, int out_channel){
    int output_size = (input_size - kernal_size)/stride + 1;

    Matrix *convM = makeMatrix(output_size * output_size, kernal_size * kernal_size, 0);
    for(int i=0; i < output_size; i++){
        for(int j=0; j < output_size; j++){
            for(int k=0; k < kernal_size; k++){
                for(int l = 0; l < kernal_size; l++){
                    if((k + i * stride < input_size) && (l + j * stride < input_size)){
                        convM->M[(i*output_size+j)*convM->col+k*kernal_size+l] = M->M[(k+i*stride)*M->col+j*stride+l];
                    }
                    else{
                        convM->M[(i*output_size+j)*convM->col+k*kernal_size+l] = 0;
                    }
                }
            }
        }
    }
    return convM;
}
Matrix* dummyMatrix(Matrix *mat){
    for(int i=0; i < mat->row * mat-> col; i++){
        float dm = i;
        mat->M[i] = dm;
    }
    return mat;
}
int main(){
    int input_size = 16;
    int kernal_size = 3;
    int stride = 2;
    int out_channel = 3;
    Matrix *A = makeMatrix(16, 16, 0);
    dummyMatrix(A);
    printMatrix(convtoMatMul(A, input_size,kernal_size, stride, out_channel));

    
}