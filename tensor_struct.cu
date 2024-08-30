#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include "tensor_struct.h"
//차원이 1과 2일 때도 생각을 해야한다. 걍 1일 때는 없는 걸로 할까......
//나중에는 나눠서 각각의 gpu 안에 넣어야하기 때문에 생각을 해보면 인덱스 값에 따라 값을 copy 해주는 것도 있으면 좋을 것 같다.
Tensor *makeTensor(int *dim, int num_dim, int device_type){
    Tensor* ten = (Tensor*)malloc(sizeof(Tensor));
    ten->dim = (int*)malloc(num_dim * sizeof(int));
    ten->stride = (int*)malloc(num_dim * sizeof(int));
    ten->num_dim = num_dim;
    ten->device_type = device_type;

    int tensorSize = 1;
    for(int i= num_dim - 1; i >= 0; i--){
        ten->dim[i] = dim[i];
        ten->stride[i] = tensorSize;
        tensorSize *= dim[i];
    }

    if(!device_type){
        ten->T = (float*)malloc(tensorSize * sizeof(float));
    }else{
        cudaSetDevice(device_type-1);
        cudaMalloc(&ten->T, tensorSize * sizeof(float));
    }
    return ten;
}
Tensor* makeTensorbyShape(Tensor* src, int device_type){
    Tensor* ten = (Tensor*)malloc(sizeof(Tensor));
    ten->dim = (int*)malloc(sizeof(int) * src->num_dim);
    ten->stride = (int*)malloc(sizeof(int) * src->num_dim);
    ten->num_dim = src->num_dim;
    ten->device_type = device_type;
    for(int i=0; i < src->num_dim; i++){
        ten->dim[i] = src->dim[i];
        ten->stride[i] = src->stride[i];
    }
    if(!device_type){
        ten->T = (float*)malloc(ten->dim[0] * ten->stride[0] * sizeof(float));
    }else{
        cudaSetDevice(device_type-1);
        cudaMalloc(&ten->T, ten->dim[0] * ten->stride[0] * sizeof(float));
    }
    return ten;
}
void freeTensor(Tensor *ten){
    if(ten==NULL){
        printf("NO TENSOR IN POINTER.\n");
        return;
    }else{
        if(ten->device_type){
            cudaSetDevice(ten->device_type - 1);
            cudaError_t err = cudaFree(ten->T);
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
            }
            
        }else{
            free(ten->T);
        }
    }
    free(ten->dim);
    free(ten->stride);
    free(ten);
}
void infoTensor(Tensor *ten){
    printf("\n=========Tensor===========\n");
    printf("DIMENTION : [");
    for(int i=0; i < ten->num_dim-1; i++){
        printf("%d ", ten->dim[i]);
    }
    printf("%d]\n", ten->dim[ten->num_dim - 1]);
    printf("DEVICE TYPE : ");
    if(ten->device_type){
        printf("GPU %d", ten->device_type);
    }else{
        printf("CPU");
    }
    printf("\n==========================\n");
}
void printTensor(Tensor *ten){
    if(ten->device_type){
        printf("printTensor : GPU mem can not be printed\n");
        return;
    }
    //==============if ten->num_dim < 3========================
    //=================else====================================
    printf("=\n");
    int numofMat = ten->dim[0] * ten->stride[0] / ten->stride[ten->num_dim-3];
    // for(int n=0; n < ten->num_dim - 2; n++){
    //     numofMat *= dim[n];
    // }
    for(int mat_inx = 0; mat_inx < numofMat; mat_inx++){
        printf("[ ");
        int tmp = mat_inx;
        for(int n=0; n < ten->num_dim - 3; n++){
            printf("%d, ", tmp / (ten->stride[n]/ten->stride[ten->num_dim-3]));
            tmp %= (ten->stride[n]/ten->stride[ten->num_dim-3]);
        }
        printf("%d, ", tmp);
        printf("-, -]\n");
        for(int i = mat_inx * ten->stride[ten->num_dim-3]; i < mat_inx * ten->stride[ten->num_dim-3] + ten->stride[ten->num_dim-3]; i+=ten->stride[ten->num_dim-2]){
            for(int j= i; j < i + ten->stride[ten->num_dim - 2];j++){
                printf("%.02f\t", ten->T[j]);
            }
            printf("\n");
        }
    }
}   

Tensor* copyTensor(Tensor *dst, Tensor *src){
    if(dst->num_dim != src->num_dim){
        printf("copyMatrix : shape of dst and src doesn't match.\n");
        return NULL;
    }
    for(int i=0; i < dst->num_dim; i++){
        if(dst->dim[i] != src->dim[i]){
            printf("copyMatrix : shape of dst and src doesn't match.\n");
            return NULL;
        }
    }
    if(!dst->device_type && !src->device_type){ //CPU to CPU
        for(int i=0; i < dst->dim[0]*dst->stride[0]; i++)
            dst->T[i] = src->T[i];
    }

    else if(dst->device_type && src->device_type){
        cudaMemcpy(dst->T, src->T, dst->dim[0]*dst->stride[0] * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    else if(dst->device_type){
        cudaSetDevice(dst->device_type -1);
        cudaMemcpy(dst->T, src->T, dst->dim[0]*dst->stride[0] * sizeof(float), cudaMemcpyHostToDevice);
    }else{
        cudaSetDevice(src->device_type -1);
        cudaMemcpy(dst->T, src->T, dst->dim[0]*dst->stride[0] * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return dst;
}

__global__ void reshape_(float* dst, float* src){
    
}


Tensor* reshapeTensor(Tensor* dst, Tensor* src, int* reshape){

    if(src->device_type != dst->device_type){
        printf("DEVICE NOT MATCH.\n");
        return NULL;
    }
    if(src->num_dim != dst->num_dim){
        printf("DEVICE NUM_DIM DOES NOT MATCH.\n");
        return NULL;
    }

    if(src->dim[0] * src->stride[0] != dst->dim[0] * dst->stride[0]){
        printf("DEVICE NUM OF ELEMENT DOES NOT MATCH.\n");
        return NULL;
    }

    //===================Setting for reshape==========================
    int* tmp_reshape = (int*)malloc(sizeof(int) * src->num_dim);
    for(int i=0; i < src->num_dim; i++){
        printf("%d ", reshape[i]);
    }
    printf("\n");
    for(int i=0; i < src->num_dim; i++){
        for(int j=0; j < src->num_dim; j++){
            if(reshape[j] == i && dst->dim[j] == src->dim[i]){//여기서 reshape이랑 맞지 않는 것도 걸러냄.
                tmp_reshape[i] = j;
                goto NEXT_RESHAPETENSOR_TMP_RESHAPE;//나도 쓰기 싫었다.
            }
        }
        printf("NOT AN APPROPRIATE RESHAPE.\n");
        return NULL;
        NEXT_RESHAPETENSOR_TMP_RESHAPE: ;
    }
    //================================================================

    for(int i=0; i < src->num_dim; i++){
        printf("%d ", tmp_reshape[i]);
    }

    printf("\n");
    if(src->device_type){
        
    }else{//CPU
        int newInx, tmp;
        for(int inx =0; inx < src->dim[0] * src->stride[0]; inx++){
            newInx = 0;
            tmp = inx;
            for(int i=0; i < src->num_dim; i++){
                newInx += tmp / src->stride[i] * dst->stride[tmp_reshape[i]];
                tmp = tmp % src->stride[i];
            }
            dst->T[newInx] = src->T[inx];
        }
    }

    free(tmp_reshape);
    return dst;
}



// Tensor* reshapeTensorinline(Tensor* ten, int* reshape){
//     //===================Setting for reshape==========================
//     int* tmp_reshape = (int*)malloc(sizeof(int) * ten->num_dim);
//     for(int i=0; i < ten->num_dim; i++){
//         printf("%d ", reshape[i]);
//     }
//     printf("\n");
//     for(int i=0; i < ten->num_dim; i++){
//         for(int j=0; j < ten->num_dim; j++){
//             if(reshape[j] == i){
//                 tmp_reshape[i] = j;
//                 goto NEXT_RESHAPETENSOR_TMP_RESHAPE;//나도 쓰기 싫었다.
//             }
//         }
//         printf("NOT AN APPROPRIATE RESHAPE.\n");
//         return NULL;
//         NEXT_RESHAPETENSOR_TMP_RESHAPE: ;
//     }
//     //===================================================


//     float* tmp;
//     for(int i=0; i < ten->num_dim; i++){
//         printf("%d ", tmp_reshape[i]);
//     }
//     printf("\n");
//     if(ten->device_type){
//         cudaSetDevice(ten->device_type-1);
//         cudaMalloc(&tmp, sizeof(float) * ten->dim[0] * ten->stride[0]);
//         <<<>>>reshape_(tmp, ten->T);

//     }else{//CPU
//         tmp_T = (float*)malloc(sizeof(float) * ten->dim[0] * ten->stride[0]);
//         int new_inx, tmp, tmp_left;
//         for(int inx=0; inx < ten->dim[0] * ten->stride[0]; inx++){
//             new_inx = 0;
//             tmp = inx;
//             for(int i=0; i < ten->num_dim; i++){
//                 tmp_left = tmp / ten->stride[i];
//                 new_inx += tmp_left * ;
                
//             }
//         }
//         free(tmp);
//     }

//     free(tmp_reshape);
//     return ten;
// }

__global__ void tiledMM(float *A, float *B, float *C, float *bias, int M, int N, int K) {
    //blockDim.z, blockIdx.z
    int matIdx = blockDim.z * blockIdx.z;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float s_a[tile_SIZE][tile_SIZE];
    __shared__ float s_b[tile_SIZE][tile_SIZE];

    float tmp = 0.0f;

    for (int i = 0; i < (K + tile_SIZE - 1) / tile_SIZE; i++) {
        if (row < M && (i * tile_SIZE + threadIdx.x) < K)
            s_a[threadIdx.y][threadIdx.x] = A[matIdx * (K*M) + row * K + (i * tile_SIZE + threadIdx.x)];
        else
            s_a[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (i * tile_SIZE + threadIdx.y) < K)
            s_b[threadIdx.y][threadIdx.x] = B[matIdx * (K*N) +(i * tile_SIZE + threadIdx.y) * N + col];
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
            C[matIdx *(M*N) + row * N + col] = tmp + bias[col];
        else
            C[matIdx *(M*N) + row * N + col] = tmp;
    }
}

void matmul_(float *dA, float *dB, float *dC, float *dBias, int M, int N, int K, int numofMat) {
    dim3 dimGrid((N + tile_SIZE - 1) / tile_SIZE, (M + tile_SIZE - 1) / tile_SIZE, numofMat); //dim은 4x3x32x32를 matmul하는 경우 12가 들어가게 된다.
    dim3 dimBlock(tile_SIZE, tile_SIZE);
    tiledMM<<<dimGrid, dimBlock>>>(dA, dB, dC, dBias, M, N, K);
}

Tensor *matmul(Tensor *dC, Tensor *dA, Tensor *dB) {
    if (dA->device_type != dB->device_type) {
        printf("two source Matrix is on different device. dA : %d, dB: %d\n", dA->device_type, dB->device_type);
        return NULL;
    }
    if (dC->device_type != dA->device_type) {
        printf("result matrix is on different device.dA : %d, dC: %d\n", dA->device_type, dC->device_type);
        return NULL;
    }
    
    if (dC->num_dim != dA->num_dim || dC->num_dim != dB->num_dim) {
        printf("matrices have different dimension.(num_dim is different)\n");
        return NULL;
    }
    for(int i=0; i < dA->num_dim-3; i++){
        if(dA->dim[i] != dB->dim[i]){
            printf("dim %d of dA and dB is not the same.\n", i);
            return NULL;
        }
    }
    if(dA->dim[dA->num_dim - 1] != dB->dim[dB->num_dim-2]){
        printf("number of column of dA and row of dB doesn't match\n");
        return NULL;
    }
    for(int i=0; i < dA->num_dim-3; i++){
        if(dA->dim[i] != dC->dim[i]){
            printf("dim %d of source and result Matrix is not the same.\n", i);
            return NULL;
        }
    }
    if(dA->dim[dA->num_dim - 2] != dC->dim[dA->num_dim - 2] || dC->dim[dC->num_dim -1] != dB->dim[dB->num_dim - 1]){
        printf("result matrix is in different dimension.\n");
        return NULL;
    }

    cudaSetDevice(dA->device_type - 1);
    matmul_(dA->T, dB->T, dC->T, NULL, dA->dim[dA->num_dim - 2], dB->dim[dB->num_dim - 1], dA->dim[dA->num_dim - 1], dA->dim[0] * dA->stride[0] / dA->stride[dA->num_dim-3]);
    return dC;
}