#include <stdio.h>
#include<stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "matrix_struct.h"

Matrix* makeMatrix(int row, int col, int device_type){
    Matrix *mat = (Matrix*)malloc(sizeof(Matrix)); 
    if(!device_type){
        mat->M = (float*)malloc(row * col * sizeof(float));
    }else{
        cudaSetDevice(device_type-1);
        cudaMalloc(&mat->M, row * col * sizeof(float));    
    }
    
    mat->device_type = device_type;
    mat->row = row;
    mat->col = col;
    return mat;
}

void freeMatrix(Matrix *mat) {
    if (mat == NULL) {
        fprintf(stderr, "Attempted to free a NULL matrix pointer\n");
        return;
    }
    if (mat->M == NULL) {
        fprintf(stderr, "Matrix data pointer is NULL\n");
    } else {
        if (mat->device_type) {
            cudaSetDevice(mat->device_type - 1);
            cudaError_t err = cudaFree(mat->M);
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
            }
        } else {
            free(mat->M);
        }
    }
    free(mat);
}


void printMatrix(Matrix *mat){
    if(mat->device_type){
        printf("printMatrix : GPU mem can not be printed\n");
        return;
    }
    printf("=\n");
    infoMatrix(mat);
    for(int i=0; i < mat->row; i++){
        for(int j=0; j < mat->col; j++){
            printf("%f\t", mat->M[i * mat->col + j]);
        }
        printf("\n");
    }
    printf("=\n");
}

void infoMatrix(Matrix *mat){
    printf("device : %d dim:(%d, %d)\n", mat->device_type,mat->row, mat->col);
}



Matrix* copyMatrix(Matrix *dst, Matrix *src){
    if(dst->row != src->row || dst->col != src->col){
        printf("copyMatrix : shape of dst and src doesn't match.\n");
        return NULL;
    }
    if(!dst->device_type && !src->device_type){ //CPU to CPU
        for(int i=0; i < dst->row * dst->col; i++)
            dst->M[i] = src->M[i];
    }
    else if(dst->device_type && src->device_type){
        cudaMemcpy(dst->M, src->M, dst->row * dst->col * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    else if(dst->device_type){
        cudaMemcpy(dst->M, src->M, dst->row * dst->col * sizeof(float), cudaMemcpyHostToDevice);
    }else{
        cudaMemcpy(dst->M, src->M, dst->row * dst->col * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return dst;
}

// Matrix* moveMatrix(Matrix *mat, int device_type){
//     cudaSetDevice(device_type-1);
//     Matrix *dMat = makeMatrix(mat->row, mat->col, device_type);
//     Matrix *tmp_mat = copyMatrix(mat, device_type);
//     if(!tmp_mat)
//         return NULL;
//     freeMatrix(mat);
//     return tmp_mat;
// }


__global__ void tiledMM(float *A, float *B, float *C, float *bias, int M, int N, int K) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float s_a[tile_SIZE][tile_SIZE];
    __shared__ float s_b[tile_SIZE][tile_SIZE];

    float tmp = 0.0f;

    for (int i = 0; i < (K + tile_SIZE - 1) / tile_SIZE; i++) {
        if (row < M && (i * tile_SIZE + threadIdx.x) < K)
            s_a[threadIdx.y][threadIdx.x] = A[row * K + (i * tile_SIZE + threadIdx.x)];
        else
            s_a[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (i * tile_SIZE + threadIdx.y) < K)
            s_b[threadIdx.y][threadIdx.x] = B[(i * tile_SIZE + threadIdx.y) * N + col];
        else
            s_b[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int j = 0; j < tile_SIZE; j++) {
            tmp += s_a[threadIdx.y][j] * s_b[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        if (bias)
            C[row * N + col] = tmp + bias[col];
        else
            C[row * N + col] = tmp;
    }
}

void matmul_(float *dA, float *dB, float *dC, float *dBias, int M, int N, int K) {
    dim3 dimGrid((N + tile_SIZE - 1) / tile_SIZE, (M + tile_SIZE - 1) / tile_SIZE);
    dim3 dimBlock(tile_SIZE, tile_SIZE);
    tiledMM<<<dimGrid, dimBlock>>>(dA, dB, dC, dBias, M, N, K);
}

Matrix *matmul_Bias(Matrix *dA, Matrix *dB, Matrix *dBias) {
    if (dA->device_type != dB->device_type || dA->device_type != dBias->device_type) {
        printf("two Matrix is on different device. dA : %d, dB: %d\n", dA->device_type, dB->device_type);
        return NULL;
    }
    if (dBias->col != dB->col) {
        printf("dBias and dB should have same num of columns\n");
        return NULL;
    }
    if (dBias->row != 1) {
        printf("dBias should have only one row.\n");
        return NULL;
    }
    if (dA->col != dB->row) {
        printf("number of column of dA and row of dB doesn't match\n");
        return NULL;
    }
    Matrix *dC = makeMatrix(dA->row, dB->col, dA->device_type);
    cudaSetDevice(dA->device_type - 1);
    matmul_(dA->M, dB->M, dC->M, dBias->M, dA->row, dB->col, dA->col);
    return dC;
}

Matrix *matmul_Bias_inline(Matrix *dC, Matrix *dA, Matrix *dB, Matrix *dBias) {
    if (dA->device_type != dB->device_type || dA->device_type != dBias->device_type) {
        printf("two Matrix is on different device. dA : %d, dB: %d\n", dA->device_type, dB->device_type);
        return NULL;
    }
    if (dBias->col != dB->col) {
        printf("dBias and dB should have same num of columns\n");
        return NULL;
    }
    if (dBias->row != 1) {
        printf("dBias should have only one row.\n");
        return NULL;
    }
    if (dA->col != dB->row) {
        printf("number of column of dA and row of dB doesn't match\n");
        return NULL;
    }
    if (dC->device_type != dA->device_type) {
        printf("result matrix is on different device.\n");
        return NULL;
    }
    if (dC->row != dA->row || dC->col != dB->col) {
        printf("result matrix is in different dimension.\n");
        return NULL;
    }
    cudaSetDevice(dA->device_type - 1);
    matmul_(dA->M, dB->M, dC->M, dBias->M, dA->row, dB->col, dA->col);
    return dC;
}

Matrix *matmul(Matrix *dA, Matrix *dB) {
    if (dA->device_type != dB->device_type) {
        printf("two Matrix is on different device. dA : %d, dB: %d\n", dA->device_type, dB->device_type);
        return NULL;
    }
    if (dA->col != dB->row) {
        printf("number of column of dA and row of dB doesn't match\n");
        return NULL;
    }
    Matrix *dC = makeMatrix(dA->row, dB->col, dA->device_type);
    cudaSetDevice(dA->device_type - 1);
    matmul_(dA->M, dB->M, dC->M, NULL, dA->row, dB->col, dA->col);
    return dC;
}

Matrix *matmul_inline(Matrix *dC, Matrix *dA, Matrix *dB) {
    if (dA->device_type != dB->device_type) {
        printf("two Matrix is on different device. dA : %d, dB: %d\n", dA->device_type, dB->device_type);
        return NULL;
    }
    if (dA->col != dB->row) {
        printf("number of column of dA and row of dB doesn't match\n");
        return NULL;
    }
    if (dC->device_type != dA->device_type) {
        printf("result matrix is on different device.\n");
        return NULL;
    }
    if (dC->row != dA->row || dC->col != dB->col) {
        printf("result matrix is in different dimension.\n");
        return NULL;
    }
    cudaSetDevice(dA->device_type - 1);
    matmul_(dA->M, dB->M, dC->M, NULL, dA->row, dB->col, dA->col);
    return dC;
}

__global__ void ReLU_device(float *dA, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(dA[i] < 0){
        dA[i] = 0;
    }
}
Matrix *ReLU_inline(Matrix *mat){//change values directly. doesn't clone.
    int thdsPerBlks = tile_SIZE*tile_SIZE;
    int numofBlks = (mat->row*mat->col+thdsPerBlks-1) / thdsPerBlks;
    cudaSetDevice(mat->device_type-1);
    ReLU_device<<<numofBlks, thdsPerBlks>>>(mat->M, mat -> row * mat -> col);
    cudaDeviceSynchronize();
    return mat;
}


__global__ void softMax(float*dRes, float *dMat, int row, int col){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < row){
        double sum = 0.0;
        for(int j=0; j < col; j++){
            sum += expf(dMat[i * col + j]);
        }
        for(int j=0; j < col; j++){
            dRes[i * col + j] = expf(dMat[i * col + j]) / sum;
        }
    }
}


Matrix *softMax_Rowwise_inline(Matrix*dRes, Matrix *dMat){
    if(dMat->row == dRes->row && dMat->col == dRes->col){
        softMax<<<((dMat->row + tile_SIZE - 1) / tile_SIZE), tile_SIZE>>>(dRes->M, dMat->M, dMat->row, dMat->col);
        return dRes;
    }else{
        printf("\"softMax_Rowwise_inline\" : src's and res's row and column is not same\n");
        return NULL;
    }
}


int isSameShape(Matrix *dMat1, Matrix *dMat2, Matrix *dMat3){
    return dMat1->row == dMat2->row && dMat1->col == dMat2->col && dMat2->row == dMat3->row && dMat2->col == dMat3-> col && dMat1-> device_type == dMat2 ->device_type &&dMat2->device_type == dMat3->device_type;
}

__global__ void matadd_(float *dMat, float *dA, float *dB, int size){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < size){
        dMat[i] = dA[i] + dB[i];
    }
}

Matrix *matAdd(Matrix *dMat, Matrix *dA, Matrix *dB){
    if(isSameShape(dMat, dA, dB)){
        int size = dA->row * dA->col;
        matadd_<<<(size+tile_SIZE*tile_SIZE -1)/tile_SIZE,tile_SIZE * tile_SIZE>>>(dMat->M, dA->M, dB->M, dA->row * dA->col);
        return dMat;
    }else{
        printf("\"matAdd\" : one of dMat, dA, dB's type does not match\n");
        return NULL;
    }
}


__global__ void matsub_(float *dMat, float *dA, float *dB, int size){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < size){
        dMat[i] = dA[i] - dB[i];
    }
}

Matrix *matSub(Matrix *dMat, Matrix *dA, Matrix *dB){
    if(isSameShape(dMat, dA, dB)){
        int size = dA->row * dA->col;
        matsub_<<<(size+tile_SIZE*tile_SIZE -1)/tile_SIZE,tile_SIZE * tile_SIZE>>>(dMat->M, dA->M, dB->M, dA->row * dA->col);
        return dMat;
    }else{
        printf("\"matAdd\" : one of dMat, dA, dB's type does not match\n"); 
        return NULL;
    }
}

// __global__ void transposeMatrix_(float* M_out, float *M, int row_size, int col_size){
//     __shared__ float s_tile[tile_SIZE][tile_SIZE+1]; // bank conflict를 피하기 위해 +1을 추가

//     int row = blockIdx.y * tile_SIZE + threadIdx.y;
//     int col = blockIdx.x * tile_SIZE + threadIdx.x;

//     // 원본 행렬의 값을 타일에 로드
//     if(row < row_size && col < col_size){
//         s_tile[threadIdx.y][threadIdx.x] = M[row * col_size + col];
//     }

//         __syncthreads();

//     // 전치된 값을 출력 행렬에 저장
//     if(row < row_size && col < col_size){
//         M_out[row * col_size + col] = s_tile[threadIdx.y][threadIdx.x];
//     }
//     // row = blockIdx.x * tile_SIZE + threadIdx.y;
//     // col = blockIdx.y * tile_SIZE + threadIdx.x;

//     // if(row < col_size && col < row_size){
//     //     M_out[row * row_size + col] = s_tile[threadIdx.x][threadIdx.y];
//     // }
// }
__global__ void transposeNaive(float *odata, float* idata, int row_size, int col_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < col_size && y < row_size) {
        odata[y * col_size + x] = idata[x * col_size + y];
    }
}

Matrix *transposeMatrix(Matrix *mat){

    float *A;
    if(mat->device_type){       //GPU
        cudaSetDevice(mat->device_type-1);
        cudaMalloc(&A, sizeof(float)* mat->col * mat->row);
        dim3 blockSize(tile_SIZE, tile_SIZE);
        dim3 gridSize((mat->col + tile_SIZE - 1) / tile_SIZE, (mat->row + tile_SIZE - 1) / tile_SIZE);
        // transposeMatrix_<<<gridSize, blockSize>>>(A, mat->M, mat->row, mat->col);
        transposeNaive<<<gridSize, blockSize>>>(A, mat->M, mat->row, mat->col);
        cudaFree(mat->M);
        mat->M = A;
    }else{                      //CPU

    }
    int tmp = mat->row;
    mat->row = mat->col;
    mat->col = tmp;
    return mat;
}

