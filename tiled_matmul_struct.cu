#include <stdio.h>
#include <cuda_runtime.h>

#define tile_SIZE 8

typedef struct Matrix{
    char device_type;
    int row;
    int col;
    float *M;
}Matrix;

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

void freeMatrix(Matrix *mat){
    if(mat->device_type){
        cudaSetDevice(mat->device_type - 1);
        cudaFree(mat->M);
    }else{
        free(mat->M);
    }
    free(mat);
}


void printMatrix(Matrix *mat){
    printf("=========%s===========\n",(mat->device_type?"GPU":"CPU"));
    for(int i=0; i < mat->row; i++){
        for(int j=0; j < mat->col; j++){
            printf("%f\t", mat->M[i * mat->col + j]);
        }
        printf("\n");
    }
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

__global__ void printGPU(float* tmp){
    printf(" printGPU : %f\n", *tmp);
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
    
    if(!device_type){//copy 할 것이 device_type일 경우
            Matrix * mat_copy = makeMatrix(mat->row, mat->col, 0);
            cudaSetDevice(mat->device_type-1);
            cudaMemcpy(mat_copy->M, mat->M, mat_copy->row * mat_copy->col * sizeof(float), cudaMemcpyDeviceToHost);
            return mat_copy;
    }else {
        int tmp;
        cudaGetDeviceCount(&tmp);
        if(device_type > 0 && device_type <= tmp){
            copyMatToDevice(mat, device_type);
        }else{
            printf("this devicetype is over the area");
            return NULL;
        }
    }return NULL;
}


__global__ void tiledMM(float *A, float *B, float *C, int M, int K){
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
    C[row * M + col] = tmp;
}


void matmul(float *dA, float *dB, float *dC, int N, int M, int K){
    if(!(N % tile_SIZE) && !(M % tile_SIZE) && !(K % tile_SIZE)){
        dim3 dimGrid(M / tile_SIZE, N / tile_SIZE);//num of blocks per grid<threadidx x==t_size,threadidx.y==t_size>
        // printf("%d, %d", N/tile_SIZE, M/tile_SIZE);
        dim3 dimBlock(tile_SIZE, tile_SIZE);//num of threads per block<threadidx x==t_size,threadidx.y==t_size>
        tiledMM<<<dimGrid, dimBlock>>>(dA, dB, dC, M, K);
    }else{
        printf("one of dA, dB, dC could not be divided by tile size.\n");
    }
}

Matrix *matmul_Struct(Matrix*dA, Matrix *dB){
    if(dA->device_type != dB->device_type){//device type 확인
        printf("two Matrix is on different device. dA : %d, dB: %d", dA->device_type, dB->device_type);
        return NULL;
    }
    Matrix *dC = makeMatrix(dA->row, dB->col, dA->device_type);
    if(dA->col == dB->row){
        cudaSetDevice(dA->device_type-1);
        matmul(dA->M, dB->M, dC->M, dA->row, dB->col, dA->col);
        return dC;
    }
    else{
        printf("column of dA and row of dB doesn't match\n");

        return NULL;
    }
}


int main(){
    int N = 16, K = 8, M = 8;
    Matrix *A, *B;
    A = makeMatrix(N, K, 0);
    B = makeMatrix(K, M, 0);
    
    for(int i=0; i < A->row * A->col; i++){
        A->M[i] = i;
    }
    for(int i=0; i < B->row * B->col; i++){
        B->M[i] = i;
    }

    Matrix *dA = copyMatToDevice(A, 2);
    Matrix *dB = copyMatToDevice(B, 2);
    freeMatrix(A);
    A = copyMatToHost(dA);
    printMatrix(A);
    printMatrix(B);
    Matrix *dC = matmul_Struct(dA, dB);

    Matrix *C = copyMatToHost(dC);
    printMatrix(C);

    freeMatrix(A);
    freeMatrix(B);
    freeMatrix(dA);
    freeMatrix(dB);
    freeMatrix(dC);
    return 0;
}
