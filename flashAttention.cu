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



__global__ void flashAttention_(float * dO, float *dQ, float *dK, float *dV, float *M, float * L, float *dM, float *dL, int num_Token, int model_Dim){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float s_a[tile_SIZE][tile_SIZE];
    __shared__ float s_b[tile_SIZE][tile_SIZE];

    float tmp = 0.0f;
    //=======================tiled Matrix Multiplication=============================
    for (int i = 0; i < (model_Dim + tile_SIZE - 1) / tile_SIZE; i++) {
        if (row < num_Token && (i * tile_SIZE + threadIdx.x) < model_Dim)
            s_a[threadIdx.y][threadIdx.x] = dQ[row * model_Dim + (i * tile_SIZE + threadIdx.x)];
        else
            s_a[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < num_Token && (i * tile_SIZE + threadIdx.y) < model_Dim)
            s_b[threadIdx.y][threadIdx.x] = dK[(i * tile_SIZE + threadIdx.y) * num_Token + col];
        else
            s_b[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int j = 0; j < tile_SIZE; j++) {
            tmp += s_a[threadIdx.y][j] * s_b[j][threadIdx.x];//scaled MM 이면 엿다가 N**(1/2) 나눠준다.
        }
        __syncthreads();
    }
    
    //임시로 넣어둔다. 여기서 부터는 s_a가 값을 가지고 있는 것이다. s_b도 이제부터 exp같은거 넣을 것이다.
    if(row < num_Token && col < num_Token)
        s_a[threadIdx.y][threadIdx.x] = tmp;
    __syncthreads();

    //==============================SOFTMAX(ROWMAX)====================================

    //shared memory lookup
    if((row == 0) && (col == 0)){//(0,0)(0,8)(8,8)(8,0)
        for(int i=0; i < tile_SIZE; i++){
            for(int j=0; j < tile_SIZE; j++){
                printf("%f\t", s_a[i][j]);
            }
            printf("\n");
        }
        printf("%d\n", gridDim.x);
    }
    __syncthreads();



    if(threadIdx.x == 0){//여기서 softMax 처리를 해줄 것이다.블록 당 하나의 쓰레드에서 연산을 하기 위해 쓰레드 y 인덱스가 0일때 리니어하게 더해준다. 왼쪽 벽만 값이 있는 타일을 생각해라.
        M[row*gridDim.x + blockIdx.x] = s_b[threadIdx.y][0];

        for(int i=1; i < tile_SIZE /*&& ((i * tile_SIZE + threadIdx.y) < model_Dim)*/; i++){
            if(M[row*gridDim.x + blockIdx.x] < s_a[threadIdx.y][i]){//tile_Max
                M[row*gridDim.x + blockIdx.x] = s_a[threadIdx.y][i];
            }
        }
        for(int i=0; i < blockDim.x; i++){
            L[row*gridDim.y + blockIdx.y] += expf(s_a[threadIdx.y][i] - M[row*gridDim.x + blockIdx.x]);    //tile_sum
        }
    }  
    __syncthreads();
    //===============================SOFTMAX(total)=========================================

    if(col==0){
        // printf("%d", gridDim.x);
        dM[row] = M[row*gridDim.x];
        dL[row] = L[row*gridDim.x];
        for(int i=1; i < gridDim.x; i++){
            if(dM[row] <M[row*gridDim.x + i]){
                dM[row] = M[row*gridDim.x + i];
            }
            dL[row] += L[row * gridDim.x + i];
        }
    }
    
    //=======================================================================================

}
Matrix* flashAttention_Naive(Matrix *dO, Matrix *dX, Matrix *wQ, Matrix *wK, Matrix *wV){
    //if()
    int num_Token = dX->row;
    int model_Dim = wQ->col;
    // int X_dim = dX->col;

    //QKV 계산
    Matrix *Q = matmul_inline(makeMatrix(dX->row, wQ->col, dX->device_type), dX, wQ);
    Matrix *K = matmul_inline(makeMatrix(dX->row, wQ->col, dX->device_type), dX, wK);
    Matrix *V = matmul_inline(makeMatrix(dX->row, wQ->col, dX->device_type), dX, wV);
    Matrix *S = matmul_inline(makeMatrix(num_Token, num_Token, dX->device_type), Q, transposeMatrix_self(K));

    softMax_Rowwise_inline(S, S);
    matmul_inline(dO, S, V);
    freeMatrix(S);
    freeMatrix(Q);
    freeMatrix(K);
    freeMatrix(V);

    return dO;
}

Matrix* flashAttention(Matrix *dO, Matrix *dX, Matrix *wQ, Matrix *wK, Matrix *wV, Matrix *dM, Matrix *dL){

    // if();
    int num_Token = dX->row;
    int model_Dim = wQ->col;
    // int X_dim = dX->col;

    //QKV 계산
    Matrix *Q = matmul_inline(makeMatrix(dX->row, wQ->col, dX->device_type), dX, wQ);
    Matrix *K = matmul_inline(makeMatrix(dX->row, wQ->col, dX->device_type), dX, wK);
    Matrix *V = matmul_inline(makeMatrix(dX->row, wQ->col, dX->device_type), dX, wV);

    //M과 L 을 위한 공간만들기 하트 뿅뿅 M[num_token][number_of_tile]
    float *M, *L;
    cudaSetDevice(dX->device_type-1);
    cudaMalloc(&M, sizeof(float) * num_Token * ((V->row + tile_SIZE - 1) / tile_SIZE));
    cudaMalloc(&L, sizeof(float) * num_Token * ((V->row + tile_SIZE - 1) / tile_SIZE));

    dim3 gridSize((V->row + tile_SIZE - 1) / tile_SIZE, (V->row + tile_SIZE - 1) / tile_SIZE);//N x N
    dim3 blockSize(tile_SIZE, tile_SIZE);
    flashAttention_<<<gridSize, blockSize>>>(dO->M, Q->M, K->M, V->M, M, L, dM->M, dL->M, num_Token, model_Dim);



    //임시 메모리 해제
    cudaFree(M);
    cudaFree(L);
    freeMatrix(Q);
    freeMatrix(K);
    freeMatrix(V);
    return dO;

}


Matrix* dummyMatrix(Matrix *mat){
    for(int i=0; i < mat->row * mat-> col; i++){
        float dm = i;
        mat->M[i] = 0.01 * dm;
    }
    return mat;
}

int main(){
    int model_dim = 8;
    int num_Token = 16;
    Matrix *wQ = dummyMatrix(makeMatrix(model_dim, model_dim, 0));
    Matrix *wK = dummyMatrix(makeMatrix(model_dim, model_dim, 0));
    Matrix *wV = dummyMatrix(makeMatrix(model_dim, model_dim, 0));
    Matrix *X = dummyMatrix(makeMatrix(num_Token, model_dim, 0));

    Matrix *dwQ = copyMatrix(makeMatrixbyShape(wQ, 1), wQ);
    Matrix *dwK = copyMatrix(makeMatrixbyShape(wK, 1), wK);
    Matrix *dwV = copyMatrix(makeMatrixbyShape(wV, 1), wV);
    Matrix *dX = copyMatrix(makeMatrixbyShape(X, 1), X);

    Matrix *dM = makeMatrix(1, num_Token, 1);
    Matrix *dL = makeMatrix(1, num_Token, 1);

    Matrix *dO = flashAttention_Naive(makeMatrix(num_Token, model_dim, 1), dX, dwQ, dwK, dwV);

    printMatrix(copyMatrix(makeMatrixbyShape(dO, 0), dO));
    // printMatrix(copyMatrix(makeMatrixbyShape(dM, 0), dM));
    // printMatrix(copyMatrix(makeMatrixbyShape(dL, 0), dL));


    // flashAttention(makeMatrix(num_Token, model_dim));
    // Matrix *Q = matmul_inline(makeMatrix(wQ->row, X->col, 1), wQ, transposeMatrix_self(dX));

}