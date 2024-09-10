#include<stdio.h>
#include<stdlib.h>
#include<float.h>
#include <curand.h>
#include <curand_kernel.h>
#include "tensor_struct_2.h"

#define MODEL_DIM 16

int random_seed = 10;

Tensor* copyTensorfromFILE(Tensor* dst, char* file_name){
    char f_name[50] = "./pre_weights/";
    for(int i=0; file_name[i]; i++){
        f_name[i+14] = file_name[i];
        f_name[i+15] = 0;
    }
    
    FILE *file = fopen(f_name, "rb");
    if (!file) {
        printf("Error opening file\n");
        return NULL;
    }

    size_t num_elements = fread(dst->T, sizeof(float), dst->dim[0]*dst->stride[0], file);
    if (num_elements != dst->dim[0]*dst->stride[0]) {
        printf("Error reading file\n");
        return NULL;
    }

    fclose(file);

    return dst;
}




Tensor* Attention_Naive(Tensor *dO, Tensor *dX, Tensor *Wqkv,Tensor* QKV, int model_dim){
    //if()
    // int num_Token = dX->dim[dX->num_dim-2];
    // int model_Dim = Wqkv->dim[dX->num_dim-1];
    // int X_dim = dX->dim[dX->num_dim-1];

    //QKV matmul
    QKV = matmul(QKV, dX, Wqkv);

    return dO;
}

int main(){
    int batch_size = 4;
    int num_head = 12;
    int num_Token = 14*14;
    int embedding_dim = 768;
    int hidden_dim = 64;

    int dim[] = {batch_size, 1, num_Token, embedding_dim};
    int qkv_weight_dim[] = {1, embedding_dim, num_head * 3 * hidden_dim};
    int QKV_dim[] = {batch_size, 1, num_Token, num_head * 3 * hidden_dim};
    
    //pretrained weight initialization
    Tensor* W = makeTensor(qkv_weight_dim, 3, 0);
    W = copyTensorfromFILE(W, "qkv_W.bin");
    printf("\n%f\n", W->T[W->dim[0]*W->stride[0]-2]);


    
    Tensor* input = makeTensor(dim, 4, 1);
    Tensor* Wqkv = makeTensor(qkv_weight_dim, 3, 1);
    Tensor* QKV = makeTensor(QKV_dim, 4, 1);

    matmul(QKV, input, Wqkv);
}


// __global__ void flashAttention_(float * dO, float *dQ, float *dK, float *dV, float *dM, float *dL, int num_Token, int model_Dim){
//     int row = blockDim.y * blockIdx.y + threadIdx.y;
//     int col = blockDim.x * blockIdx.x + threadIdx.x;

//     //지속적으로 tile 안에서 max와 sum을 저장하는 값이다.(왜냐하면 thread의 개수는 tile로 되어있는데 row 단위로 세야하니까 각 block마다 하나씩 필요하다.)
//     __shared__ float s_M[tile_SIZE];
//     __shared__ float s_L[tile_SIZE];
//     __shared__ float s_tmp_M[tile_SIZE];
//     __shared__ float s_tmp_L[tile_SIZE];

//     if(threadIdx.x < tile_SIZE) {
//         s_M[threadIdx.x] = 0.0f;
//         s_L[threadIdx.x] = 0.0f;
//     }
//     __syncthreads(); // Ensure all threads have initialized shared memory


//     //s_Q와 s_K를 넣는 공간. Modeldim 만큼의 공간이 필요한 이유는 Q든 K든 하여간 한꺼번에 가둬두고 계속 봐야하는데 tile_size 단위로 짜르면 계속 못보게 된다. 특히 s_Q는 계속 써야함.
//     __shared__ float s_Q[tile_SIZE][MODEL_DIM];
//     __shared__ float s_K[MODEL_DIM][tile_SIZE];
//     __shared__ float s_V[tile_SIZE][MODEL_DIM];
//     __shared__ float s_O[tile_SIZE][MODEL_DIM];

//     float tmp = 0.0f;

//     //=======================tiled Matrix Multiplication=============================
//     //key iteration

//     //Q to shared -> 이걸 할 때 for문을 써야하는데 이 이유는 기존 사이즈가 같은 tilesize만큼 줬기 때문에 그냥 if문으로만 처리하면 되었지만, 지금은 MODEL_DIM 에 맞춰서 메모리를 넣어줘야하기 때문에 이렇다.
//     for(int column_Q = 0; column_Q < (MODEL_DIM + tile_SIZE - 1) / tile_SIZE; column_Q++) {
//         int global_col = column_Q * tile_SIZE + threadIdx.x;
//         if(row < num_Token && global_col < MODEL_DIM) {
//             s_Q[threadIdx.y][global_col] = dQ[row * MODEL_DIM + global_col];
//         } else {
//             s_Q[threadIdx.y][global_col] = 0.0f;
//         }

//         __syncthreads();
//     }

//     // Load K to shared memory
//     // for (int row_K = 0; row_K < (MODEL_DIM + tile_SIZE - 1) / tile_SIZE; row_K++) {
        
//     //     int global_row = row_K * tile_SIZE + threadIdx.y;
//     //     if (col < num_Token && global_row < MODEL_DIM) {
//     //         s_K[global_row][threadIdx.x] = dK[global_row * num_Token + col];
//     //     } else {
//     //         s_K[global_row][threadIdx.x] = 0.0f;
//     //     }

//     //     __syncthreads();
//     // }
    
//     //K-iteration
//     for(int k_it = 0; k_it < (num_Token + tile_SIZE - 1) / tile_SIZE; k_it++){

//         tmp = 0.0f;

//         //load K to shared => 이건 K가 k_iteration마다 한 tile column
//         for(int row_K = 0; row_K < (MODEL_DIM + tile_SIZE - 1) / tile_SIZE; row_K++) {
            
//             int global_row = row_K * tile_SIZE + threadIdx.y;

//             if (col < num_Token && global_row < MODEL_DIM) {
//                 s_K[global_row][threadIdx.x] = dK[global_row * num_Token + (k_it * tile_SIZE + col)];
//             } else {
//                 s_K[global_row][threadIdx.x] = 0.0f;
//             }

//             __syncthreads();
//         }

//         //QK tiled matrix mult
//         for(int i=0; i < MODEL_DIM; i++){
//             tmp += s_Q[threadIdx.y][i] * s_K[i][threadIdx.x];
//         }
//         __syncthreads();


//         //masking with tmp
//         //================


//         //result of tmp to s_K
//         s_K[threadIdx.y][threadIdx.x] = tmp;
//         __syncthreads();
        
//         // //s_K rowMax
//         // if(col == 0){
//         //     int max = s_K[threadIdx.y][0];
//         //     for(int i=1; i < tile_SIZE; i++){
//         //         if(max < s_K[threadIdx.y][i])
//         //             max = s_K[threadIdx.y][i];
//         //     }
//         //     s_tmp_M[threadIdx.y] = max;
//         // }
//         // __syncthreads();
//         // if(col ==1){
//         //     printf("%f");
//         // }
//         // //max값에 기존 tmp 값을 빼주고 exp를 취한다.
//         // tmp = expf(tmp - s_tmp_M[threadIdx.y]);

//         // __syncthreads();
        
//         // if(col == 0){
//         //     int sum;
//         // }

//         for(int column_Q = 0; column_Q < (MODEL_DIM + tile_SIZE - 1) / tile_SIZE; column_Q++) {
//             int global_col = column_Q * tile_SIZE + threadIdx.x;
//             if(row < num_Token && global_col < MODEL_DIM) {
//                 s_Q[threadIdx.y][global_col] = dQ[row * MODEL_DIM + global_col];
//             } else {
//                 s_Q[threadIdx.y][global_col] = 0.0f;
//             }

//             __syncthreads();
//         }


//     }


//     //     // shared memory lookup
//     // if ((row == 16) && (col == 0)) { // Printing only for thread (0,0)
//     //     printf("dQ data:\n");
//     //     for (int i = 0; i < tile_SIZE; i++) {
//     //         for (int j = 0; j < MODEL_DIM; j++) {
//     //             printf("%f\t", dQ[(i + blockIdx.y * blockDim.y) * MODEL_DIM + j]);
//     //         }
//     //         printf("\n");
//     //     }
//     //     printf("\n");

//     //     printf("s_Q data:\n");
//     //     for (int i = 0; i < tile_SIZE; i++) {
//     //         for (int j = 0; j < MODEL_DIM; j++) {
//     //             printf("%f\t", s_Q[i][j]);
//     //         }
//     //         printf("\n");
//     //     }
//     //     printf("\n");
//     //     __syncthreads();

//     // // shared memory lookup for s_K
//     //     printf("dK data:\n");
//     //     for (int i = 0; i < MODEL_DIM; i++) {
//     //         for (int j = 0; j < tile_SIZE; j++) {
//     //             printf("%f\t", dK[i * num_Token +(j + blockIdx.y * blockDim.y)]);
//     //         }
//     //         printf("\n");
//     //     }
//     //     printf("\n%d %d\n", blockIdx.y, blockIdx.x);
        
//     //     printf("s_K data:\n");
//     //     for (int i = 0; i < MODEL_DIM; i++) {
//     //         for (int j = 0; j < tile_SIZE; j++) {
//     //             printf("%f\t", s_K[i][j]);
//     //         }
//     //         printf("\n");
//     //     }
//     //     printf("\n");
//     // }






//     // for (int i = 0; i < (model_Dim + tile_SIZE - 1) / tile_SIZE; i++) {
//     //     if (row < num_Token && (i * tile_SIZE + threadIdx.x) < model_Dim)
//     //         s_a[threadIdx.y][threadIdx.x] = dQ[row * model_Dim + (i * tile_SIZE + threadIdx.x)];
//     //     else
//     //         s_a[threadIdx.y][threadIdx.x] = 0.0f;

//     //     if (col < num_Token && (i * tile_SIZE + threadIdx.y) < model_Dim)
//     //         s_b[threadIdx.y][threadIdx.x] = dK[(i * tile_SIZE + threadIdx.y) * num_Token + col];
//     //     else
//     //         s_b[threadIdx.y][threadIdx.x] = 0.0f;

//     //     __syncthreads();

//     //     for (int j = 0; j < tile_SIZE; j++) {
//     //         tmp += s_a[threadIdx.y][j] * s_b[j][threadIdx.x];//scaled MM 이면 엿다가 N**(1/2) 나눠준다.
//     //     }
//     //     __syncthreads();
//     // }
    
//     //==================================random seed================================


// }



// Matrix* flashAttention(Matrix *dO, Matrix *dX, Matrix *wQ, Matrix *wK, Matrix *wV, Matrix *dM, Matrix *dL){

//     // if();
//     int num_Token = dX->row;
//     int model_Dim = wQ->col;
//     // int X_dim = dX->col;

//     //QKV 계산
//     Matrix *Q = matmul_inline(makeMatrix(dX->row, wQ->col, dX->device_type), dX, wQ);
//     Matrix *K = transposeMatrix_self(matmul_inline(makeMatrix(dX->row, wQ->col, dX->device_type), dX, wK));
//     Matrix *V = matmul_inline(makeMatrix(dX->row, wQ->col, dX->device_type), dX, wV);


//     dim3 gridSize(1, (num_Token + tile_SIZE - 1) / tile_SIZE);//N x 1
//     dim3 blockSize(tile_SIZE, tile_SIZE);
//     flashAttention_<<<gridSize, blockSize>>>(dO->M, Q->M, K->M, V->M, dM->M, dL->M, num_Token, model_Dim);


    

//     //임시 메모리 해제
//     freeMatrix(Q);
//     freeMatrix(K);
//     freeMatrix(V);
//     return dO;

//     //M과 L 을 위한 공간만들기 하트 뿅뿅 M[num_token][number_of_tile]
//     // float *M, *L;
//     // cudaSetDevice(dX->device_type-1);
//     // cudaMalloc(&M, sizeof(float) * num_Token * ((V->row + tile_SIZE - 1) / tile_SIZE));
//     // cudaMalloc(&L, sizeof(float) * num_Token * ((V->row + tile_SIZE - 1) / tile_SIZE));
// }


// Matrix* dummyMatrix(Matrix *mat){
//     for(int i=0; i < mat->row * mat-> col; i++){
//         float dm = i;
//         mat->M[i] = 0.01 * dm;
//     }
//     return mat;
// }

// int main(){
//     int model_dim = MODEL_DIM;
//     int num_Token = 32;
//     Matrix *wQ = dummyMatrix(makeMatrix(model_dim, model_dim, 0));
//     Matrix *wK = dummyMatrix(makeMatrix(model_dim, model_dim, 0));
//     Matrix *wV = dummyMatrix(makeMatrix(model_dim, model_dim, 0));
//     Matrix *X = dummyMatrix(makeMatrix(num_Token, model_dim, 0));
//     Matrix *dwQ = copyMatrix(makeMatrixbyShape(wQ, 1), wQ);
//     Matrix *dwK = copyMatrix(makeMatrixbyShape(wK, 1), wK);
//     Matrix *dwV = copyMatrix(makeMatrixbyShape(wV, 1), wV);
//     Matrix *dX = copyMatrix(makeMatrixbyShape(X, 1), X);

//     Matrix *dM = makeMatrix(1, num_Token, 1);
//     Matrix *dL = makeMatrix(1, num_Token, 1);
//     Matrix *dS = makeMatrix(num_Token, num_Token, 1);

//     // flashTest(dS, dX, dwQ, dwK, dwV);
//     // Matrix *dO = flashAttention_Naive(makeMatrix(num_Token, model_dim, 1), dX, dwQ, dwK, dwV);
//     // printMatrix(copyMatrix(makeMatrixbyShape(dO, 0), dO));
    
//     Matrix *dO = flashAttention(makeMatrix(num_Token, model_dim, 1), dX, dwQ, dwK, dwV, dM, dL);

//     // printMatrix(copyMatrix(makeMatrixbyShape(dS, 0), dS));
//     // printMatrix(copyMatrix(makeMatrixbyShape(dM, 0), dM));
//     // printMatrix(copyMatrix(makeMatrixbyShape(dL, 0), dL));


//     // flashAttention(makeMatrix(num_Token, model_dim));
//     // Matrix *Q = matmul_inline(makeMatrix(wQ->row, X->col, 1), wQ, transposeMatrix_self(dX));

// }



// //================================쓰레기 통======================================

// // Matrix* flashTest(Matrix *dS, Matrix *dX, Matrix *wQ, Matrix *wK, Matrix *wV){
// //     //if()
// //     int num_Token = dX->row;
// //     int model_Dim = wQ->col;
// //     int X_dim = dX->col;

// //     //QKV 계산
// //     Matrix *Q = matmul_inline(makeMatrix(num_Token, model_Dim, dX->device_type), dX, wQ);
// //     Matrix *K = matmul_inline(makeMatrix(num_Token, model_Dim, dX->device_type), dX, wK);
// //     Matrix *V = matmul_inline(makeMatrix(num_Token, model_Dim, dX->device_type), dX, wV);
// //     transposeMatrix_self(K);
// //     printMatrix(copyMatrix(makeMatrixbyShape(K, 0), K));
// //     dim3 gridSize((num_Token + tile_SIZE - 1) / tile_SIZE, (num_Token + tile_SIZE - 1) / tile_SIZE);//N x N
// //     dim3 blockSize(tile_SIZE, tile_SIZE);
// //     flashAttention__<<<gridSize, blockSize>>>(dS->M, Q->M, K->M, V->M, num_Token, model_Dim);//num_Token : N M, model_Dim : K
// //     freeMatrix(Q);
// //     freeMatrix(K);
// //     freeMatrix(V);
// // }


// // __global__ void flashAttention__(float *dS, float *dQ, float *dK, float *dV, int num_Token, int model_Dim){
// //     int row = blockDim.y * blockIdx.y + threadIdx.y;
// //     int col = blockDim.x * blockIdx.x + threadIdx.x;

// //     __shared__ float s_a[tile_SIZE][tile_SIZE];
// //     __shared__ float s_b[tile_SIZE][tile_SIZE];

// //     float tmp = 0.0f;
// //     //=======================tiled Matrix Multiplication=============================
// //     for (int i = 0; i < (model_Dim + tile_SIZE - 1) / tile_SIZE; i++) {
// //         if (row < num_Token && (i * tile_SIZE + threadIdx.x) < model_Dim)
// //             s_a[threadIdx.y][threadIdx.x] = dQ[row * model_Dim + (i * tile_SIZE + threadIdx.x)];
// //         else
// //             s_a[threadIdx.y][threadIdx.x] = 0.0f;

// //         if (col < num_Token && (i * tile_SIZE + threadIdx.y) < model_Dim)
// //             s_b[threadIdx.y][threadIdx.x] = dK[(i * tile_SIZE + threadIdx.y) * num_Token + col];
// //         else
// //             s_b[threadIdx.y][threadIdx.x] = 0.0f;

// //         __syncthreads();

// //         for (int j = 0; j < tile_SIZE; j++) {
// //             tmp += s_a[threadIdx.y][j] * s_b[j][threadIdx.x];//scaled MM 이면 엿다가 N**(1/2) 나눠준다.
// //         }
// //         __syncthreads();
// //     }
    
// //     //임시로 넣어둔다. 여기서 부터는 s_a가 값을 가지고 있는 것이다. s_b도 이제부터 exp같은거 넣을 것이다.

// //     if(row < num_Token && col < num_Token)
// //         s_a[threadIdx.y][threadIdx.x] = tmp;

// //     __syncthreads();

// //     //==============================SOFTMAX(ROWMAX)====================================
// //     if (row < num_Token && col < num_Token)
// //         dS[row * num_Token + col] = s_a[threadIdx.y][threadIdx.x];
// //     //shared memory lookup
// //     if((row == 0) && (col == 0)){//(0,0)(0,8)(8,8)(8,0)
// //         for(int i=0; i < tile_SIZE; i++){
// //             for(int j=0; j < tile_SIZE; j++){
// //                 printf("%f\t", s_a[i][j]);
// //                 dS[(row + i)* num_Token + col + j] = s_a[i][j];
// //             }
// //             printf("\n");
// //         }
// //         printf("%d\n", gridDim.x);
// //     }

// //     __syncthreads();

// // }