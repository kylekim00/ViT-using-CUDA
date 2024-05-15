#include <stdio.h>
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
        printf("GPU mem can not be printed");
        return;
    }
    for(int i=0; i < mat->row; i++){
        for(int j=0; j < mat->col; j++){
            printf("%f\t", mat->M[i * mat->col + j]);
        }
        printf("\n");
    }
}

void infoMatrix(Matrix *mat){
    printf("device : %d dim:(%d, %d)\n", mat->device_type,mat->row, mat->col);
}

Matrix* copyMatToDevice(Matrix *mat, int device_type){
    if(!mat->device_type){
        cudaSetDevice(device_type-1);
        Matrix *dMat = makeMatrix(mat->row, mat->col, device_type);
        cudaMemcpy(dMat->M, mat->M, mat->row * mat->col * sizeof(float), cudaMemcpyHostToDevice);
        return dMat;
    }
    return NULL;
}

Matrix* copyMatToHost(Matrix *dMat){
    if(dMat->device_type){
        Matrix * mat = makeMatrix(dMat->row, dMat->col, 0);
        cudaSetDevice(dMat->device_type-1);
        cudaMemcpy(mat->M, dMat->M, mat->row * mat->col * sizeof(float), cudaMemcpyDeviceToHost);
        return mat;
    }
    return NULL;
}

Matrix* copyMatrix(Matrix *mat, int device_type){
    if(!device_type){//device_type == cpu
            Matrix * mat_copy = makeMatrix(mat->row, mat->col, 0);
            cudaSetDevice(mat->device_type-1);
            cudaMemcpy(mat_copy->M, mat->M, mat_copy->row * mat_copy->col * sizeof(float), cudaMemcpyDeviceToHost);
            return mat_copy;
    }else {
        if(!mat->device_type){
            int tmp;
            cudaGetDeviceCount(&tmp);
            if(device_type > 0 && device_type <= tmp){
                return copyMatToDevice(mat, device_type);
            }else{
                printf("invalid device type\n");
                return NULL;
            }
        }else{
            //여기에 devicetodevice를 조진다.
            printf("invalid device type\n");
            return NULL;
        }
    }
}

Matrix* moveMatrix(Matrix *mat, int device_type){
    Matrix *tmp_mat = copyMatrix(mat, device_type);
    if(!tmp_mat)
        return NULL;
    freeMatrix(mat);
    return tmp_mat;
}

__global__ void tiledMM(float *A, float *B, float *C,float *bias, int M, int K){
    // row, col : 얘네들은 쓰레드의 위치를 나타내니까 쓰레드의 구분이 연산과정에서 필요할 때 써주면 된다.
    // threadIdx.y, threadIdx.x : 얘네들은 블럭 크기의 메모리를 다루어줄 때 쓰면된다. 차이점은 block을 구분지어야 할 필요가 있는 cuda 연산에서는 block을 붙여줘야하고
    // shared memory 와 같이 그냥 안에서 일어나는 것은  idx로 해줘도 된다는 것이다.
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float s_a[tile_SIZE][tile_SIZE];
    __shared__ float s_b[tile_SIZE][tile_SIZE];

    int tmp = 0;

    for(int i=0; i < K; i += tile_SIZE){//i는 tile_size만큼 늘어난다.....
        //값을 가져올 때 A는 rowtile을 기준으로 C의 row를 기준으로 가져와야하고 B는 column tile을 기준으로 들고 와야하기 때문에 
        s_a[threadIdx.y][threadIdx.x] = A[row * K + (i + threadIdx.x)];
        s_b[threadIdx.y][threadIdx.x] = B[(i + threadIdx.y) * M + col]; // 먼저 i * N으로 타일에 포지션을 잡고, threadidx.y는 타일의 y 인덱스를 가리킨다. 
        __syncthreads();
        // printf("A: bidx : (%d,%d), tidx : (%d,%d) | %d\n",blockIdx.y,blockIdx.x,threadIdx.y,threadIdx.x, s_a[threadIdx.y][threadIdx.x]);
        // __syncthreads();
        // printf("B: bidx : (%d,%d), tidx : (%d,%d) | %d\n",blockIdx.y,blockIdx.x,threadIdx.y,threadIdx.x, s_b[threadIdx.y][threadIdx.x]);
        // 이제 계산한다. 
        for(int j=0; j < tile_SIZE; j++){
            tmp += s_a[threadIdx.y][j] * s_b[j][threadIdx.x];
        }

        __syncthreads();
    }
    // printf("(%d, %d) %d\n", row, col, tmp);
    if(bias)
        C[row * M + col] = tmp + bias[col];
    else
        C[row * M + col] = tmp;
}

// __global__ void matMul_Naive(int *A, int *B, int *C, int n){
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if(i < n*n){
//         int x = i / n;
//         int y = i % n;
//         int sum = 0;
//         for(int k=0; k < n; k++){
//             sum += A[x*n + k] * B[k*n + y];
//         }
//         C[i] = sum;
//     }
// }


void matmul_(float *dA, float *dB, float *dC,float *dBias, int N, int M, int K){
    if(!(N % tile_SIZE) && !(M % tile_SIZE) && !(K % tile_SIZE)){
        dim3 dimGrid(M / tile_SIZE, N / tile_SIZE);//num of blocks per grid<threadidx x==t_size,threadidx.y==t_size>
        // printf("%d, %d", N/tile_SIZE, M/tile_SIZE);
        dim3 dimBlock(tile_SIZE, tile_SIZE);//num of threads per block<threadidx x==t_size,threadidx.y==t_size>
        tiledMM<<<dimGrid, dimBlock>>>(dA, dB, dC,dBias, M, K);
    }else{
        printf("one of dA, dB, dC could not be divided by tile size.\n");
        
    }
}

Matrix *matmul_Bias(Matrix*dA, Matrix *dB, Matrix *dBias){
    if(dA->device_type != dB->device_type || dA->device_type != dBias->device_type){//device type 확인
        printf("two Matrix is on different device. dA : %d, dB: %d", dA->device_type, dB->device_type);
        return NULL;
    }
    if(dBias->col != dB->col){
        printf("dBias and dB should have same num of columns\n");
        return NULL;
    }
    if(dBias->row != 1){
        printf("dBias should have only one row.\n");
        return NULL;
    }
    if(dA->col != dB->row){
        printf("number of column of dA and row of dB doesn't match\n");
        return NULL;
    }
    Matrix *dC = makeMatrix(dA->row, dB->col, dA->device_type);
    cudaSetDevice(dA->device_type-1);
    matmul_(dA->M, dB->M, dC->M, dBias->M, dA->row, dB->col, dA->col);
    return dC;
    
}

Matrix *matmul_Bias_inline(Matrix*dC, Matrix*dA, Matrix *dB, Matrix *dBias){
    if(dA->device_type != dB->device_type || dA->device_type != dBias->device_type){//device type 확인
        printf("two Matrix is on different device. dA : %d, dB: %d", dA->device_type, dB->device_type);
        return NULL;
    }
    if(dBias->col != dB->col){
        printf("dBias and dB should have same num of columns\n");
        return NULL;
    }
    if(dBias->row != 1){
        printf("dBias should have only one row.\n");
        return NULL;
    }
    if(dA->col != dB->row){
        printf("number of column of dA and row of dB doesn't match\n");
        return NULL;
    }
    if(dC ->device_type != dA->device_type){
        printf("result matrix is on different device.\n");
        return NULL;
    }
    if(dC->row != dA->row || dC->col != dB->col){
        printf("result matrix is in different dimension.\n");
        return NULL;
    }
    cudaSetDevice(dA->device_type-1);
    matmul_(dA->M, dB->M, dC->M, dBias->M, dA->row, dB->col, dA->col);
    return dC;
    
}


Matrix *matmul(Matrix*dA, Matrix *dB){
    if(dA->device_type != dB->device_type){//device type 확인
        printf("two Matrix is on different device. dA : %d, dB: %d", dA->device_type, dB->device_type);
        return NULL;
    }
    if(dA->col != dB->row){
        printf("number of column of dA and row of dB doesn't match\n");
        return NULL;
    }
    Matrix *dC = makeMatrix(dA->row, dB->col, dA->device_type);
    cudaSetDevice(dA->device_type-1);
    matmul_(dA->M, dB->M, dC->M,NULL, dA->row, dB->col, dA->col);
    return dC;
}

Matrix *matmul_inline(Matrix*dC, Matrix*dA, Matrix *dB){
    if(dA->device_type != dB->device_type){//device type 확인
        printf("two Matrix is on different device. dA : %d, dB: %d", dA->device_type, dB->device_type);
        return NULL;
    }
    if(dA->col != dB->row){
        printf("number of column of dA and row of dB doesn't match\n");
        return NULL;
    }
    if(dC ->device_type != dA->device_type){
        printf("result matrix is on different device.\n");
        return NULL;
    }
    if(dC->row != dA->row || dC->col != dB->col){
        printf("result matrix is in different dimension.\n");
        return NULL;
    }
    cudaSetDevice(dA->device_type-1);
    matmul_(dA->M, dB->M, dC->M, NULL, dA->row, dB->col, dA->col);
    return dC;
}

__global__ void ReLU_device(float *dA, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(dA[i] < 0){
        dA[i] = 0;
    }
}
Matrix *ReLU(Matrix *mat){//change values directly. doesn't clone.
    int thdsPerBlks = tile_SIZE*tile_SIZE;
    int numofBlks = (mat->row*mat->col+thdsPerBlks-1) / thdsPerBlks;
    cudaSetDevice(mat->device_type-1);
    ReLU_device<<<numofBlks, thdsPerBlks>>>(mat->M, mat -> row * mat -> col);
    cudaDeviceSynchronize();
    return mat;
}


// __global__ void softMax(float *dA, int size){
//     expf(*dA);
// }


// Matrix *softMax_Rowwise_inline(Matrix *dMat){
//     softMax<<<dMat->row, dMat->col>>>(dMat->M);

// }

Matrix *softMax_Rowwise_inline(Matrix *res_Mat, Matrix *mat){
    for(int i=0; i < mat->row; i++){
        double tmp = 0.0;
        for(int j=0; j < mat -> col; j++){
            tmp += exp(mat->M[i * mat->col + j]);
        }
        for(int j=0; j < mat -> col; j++){
            res_Mat->M[i * mat->col + j] = exp(mat->M[i * mat->col + j]) / tmp;
        }
    }
    return res_Mat;
}

