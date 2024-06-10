#include<stdio.h>
#include<stdlib.h>
#include<float.h>
#include<matrix_struct.h>

// typedef struct Matrix{
//     float *M;
//     int row;
//     int col;
// }Matrix;
#define tile_SIZE 8
__global__ void flashAttention(float *dQ, float *dK, float *dV,float * dO, int N, int K, int M){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float s_a[tile_SIZE][tile_SIZE];
    __shared__ float s_b[tile_SIZE][tile_SIZE];

    float tmp = 0.0f;
    //=======================tiled Matrix Multiplication=============================
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
            tmp += s_a[threadIdx.y][j] * s_b[j][threadIdx.x];//scaled MM 이면 엿다가 N**(1/2) 나눠준다.
        }
        __syncthreads();
    }
    
    //임시로 넣어둔다. 여기서 부터는 s_a가 값을 가지고 있는 것이다. s_b도 이제부터 exp같은거 넣을 것이다.
    s_a[threadIdx.y][threadIdx.x] = tmp;
    __syncthreads();

    //==============================SOFTMAX(ROWMAX)====================================
    
    if(threadIdx.x){//여기서 SOFTMax 처리를 해줄 것이다.블록 당 하나의 쓰레드에서 연산을 하기 위해 쓰레드 y 인덱스가 0일때 리니어하게 더해준다. 왼쪽 벽만 값이 있는 타일을 생각해라.
        tmp = 0;
        for(int i=0; i < blockDim.x; i++){
            s_b[threadIdx.y][0] = expf(s_a[threadIdx.y][i]);
            tmp += s_b[threadIdx.y][0];
        }
        
    }
    
    // if(i < row){
    //     float sum = 0.0;
    //     float max = -__FLT_MAX__;
    //     for(int j=0; j < col; j++){
    //         if(max < dMat[i*col + j]) max = dMat[i*col+j];
    //     }
    //     for(int j=0; j < col; j++){
    //         sum += expf(dMat[i * col + j]-max);
    //     }
    //     for(int j=0; j < col; j++){
    //         dRes[i * col + j] = expf(dMat[i * col + j]-max) / sum;
    //     }
    // }
}
+

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
    printf("\nmin:%lf", -(__DBL_MAX__));
    
}