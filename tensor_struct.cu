#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include "tensor_struct.h"
//차원이 1과 2일 때도 생각을 해야한다. 걍 1일 때는 없는 걸로 할까......
//나중에는 나눠서 각각의 gpu 안에 넣어야하기 때문에 생각을 해보면 인덱스 값에 따라 값을 copy 해주는 것도 있으면 좋을 것 같다.

Tensor *makeTensor(int *dim, int num_dim, int device_type){
    if(num_dim < 3){
        return NULL;
    }
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


// float* makeTensorwithNewSpace(Tensor* ten){
//     float* tmp = ten->T;
//     ten->T = 
// }



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
    //==============if ten->num_dim < 3========================ls
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
    if(src->device_type){//GPU
        cudaSetDevice(src->device_type - 1);
        
        
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



// __global__ void tiledMM(float *A, float *B, float *C, float *bias, int M, int N, int K) {
//     //blockDim.z, blockIdx.z 이게 매트릭스의 수//z thread를 늘린다고 문제가 해결되지 않는다.
    
//     int matIdx = blockDim.z * blockIdx.z;
//     int row = blockDim.y * blockIdx.y + threadIdx.y;
//     int col = blockDim.x * blockIdx.x + threadIdx.x;
//     __shared__ float s_a[tile_SIZE][tile_SIZE];
//     __shared__ float s_b[tile_SIZE][tile_SIZE];

//     float tmp = 0.0f;

//     for (int i = 0; i < (K + tile_SIZE - 1) / tile_SIZE; i++) {
//         if (row < M && (i * tile_SIZE + threadIdx.x) < K)
//         //A와 B에 데이터를 넣을 때 matIdx에 차이를 두어야 한다.
//         //예를 들자면 matIdxA = blockDim.z * blockIdx.z / big_Dim_stride; 이러면 반복이 되니까.
//             s_a[threadIdx.y][threadIdx.x] = A[matIdx * (K*M) + row * K + (i * tile_SIZE + threadIdx.x)];
//         else
//             s_a[threadIdx.y][threadIdx.x] = 0.0f;

//         if (col < N && (i * tile_SIZE + threadIdx.y) < K)
//             s_b[threadIdx.y][threadIdx.x] = B[matIdx * (K*N) +(i * tile_SIZE + threadIdx.y) * N + col];
//         else
//             s_b[threadIdx.y][threadIdx.x] = 0.0f;

//         __syncthreads();

//         for (int j = 0; j < tile_SIZE; j++) {
//             tmp += s_a[threadIdx.y][j] * s_b[j][threadIdx.x];
//         }

//         __syncthreads();
//     }

//     if (row < M && col < N) {
//         if (bias)
//             C[matIdx *(M*N) + row * N + col] = tmp + bias[col];
//         else
//             C[matIdx *(M*N) + row * N + col] = tmp;
//     }
// }

__global__ void tiledMM(float *A, float *B, float *C, float *bias, int M, int N, int K, int big_dim_stride, int big_A_True) {
    //blockDim.z, blockIdx.z 이게 매트릭스의 수//z thread를 늘린다고 문제가 해결되지 않는다.
    int matIdx_A, matIdx_B;
    if(big_A_True){
        matIdx_A = blockDim.z * blockIdx.z;
        matIdx_B = matIdx_A % big_dim_stride;
    }else{
        matIdx_B = blockDim.z * blockDim.z;
        matIdx_A = matIdx_B % big_dim_stride;
    }
    
    
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float s_a[tile_SIZE][tile_SIZE];
    __shared__ float s_b[tile_SIZE][tile_SIZE];

    float tmp = 0.0f;

    for (int i = 0; i < (K + tile_SIZE - 1) / tile_SIZE; i++) {
        if (row < M && (i * tile_SIZE + threadIdx.x) < K)
        //A와 B에 데이터를 넣을 때 matIdx에 차이를 두어야 한다.
        //예를 들자면 matIdxA = blockDim.z * blockIdx.z / big_Dim_stride; 이러면 반복이 되니까.
            s_a[threadIdx.y][threadIdx.x] = A[matIdx_A * (K*M) + row * K + (i * tile_SIZE + threadIdx.x)];
        else
            s_a[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (i * tile_SIZE + threadIdx.y) < K)
            s_b[threadIdx.y][threadIdx.x] = B[matIdx_B * (K*N) +(i * tile_SIZE + threadIdx.y) * N + col];
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
            tmp = tmp + bias[col];
        if(big_A_True)
            C[matIdx_A *(M*N) + row * N + col] = tmp;
        else
            C[matIdx_B *(M*N) + row * N + col] = tmp;
    }
}


// void matmul_(float *dA, float *dB, float *dC, float *dBias, int M, int N, int K, int numofMat) {
//     dim3 dimGrid((N + tile_SIZE - 1) / tile_SIZE, (M + tile_SIZE - 1) / tile_SIZE, numofMat); //dim은 4x3x32x32를 matmul하는 경우 12가 들어가게 된다.
//     dim3 dimBlock(tile_SIZE, tile_SIZE);
//     tiledMM<<<dimGrid, dimBlock>>>(dA, dB, dC, dBias, M, N, K);
// }

// Tensor *matmul(Tensor *dC, Tensor *dA, Tensor *dB) {
//     if (dA->device_type != dB->device_type) {
//         printf("two source Matrix is on different device. dA : %d, dB: %d\n", dA->device_type, dB->device_type);
//         return NULL;
//     }
//     if (dC->device_type != dA->device_type) {
//         printf("result matrix is on different device.dA : %d, dC: %d\n", dA->device_type, dC->device_type);
//         return NULL;
//     }
    
//     if (dC->num_dim != dA->num_dim || dC->num_dim != dB->num_dim) {
//         printf("matrices have different dimension.(num_dim is different)\n");
//         return NULL;
//     }
//     for(int i=0; i < dA->num_dim-3; i++){
//         if(dA->dim[i] != dB->dim[i]){
//             printf("dim %d of dA and dB is not the same.\n", i);
//             return NULL;
//         }
//     }
//     if(dA->dim[dA->num_dim - 1] != dB->dim[dB->num_dim-2]){
//         printf("number of column of dA and row of dB doesn't match\n");
//         return NULL;
//     }
//     for(int i=0; i < dA->num_dim-3; i++){
//         if(dA->dim[i] != dC->dim[i]){
//             printf("dim %d of source and result Matrix is not the same.\n", i);
//             return NULL;
//         }
//     }
//     if(dA->dim[dA->num_dim - 2] != dC->dim[dA->num_dim - 2] || dC->dim[dC->num_dim -1] != dB->dim[dB->num_dim - 1]){
//         printf("result matrix is in different dimension.\n");
//         return NULL;
//     }

//     cudaSetDevice(dA->device_type - 1);
//     matmul_(dA->T, dB->T, dC->T, NULL, dA->dim[dA->num_dim - 2], dB->dim[dB->num_dim - 1], dA->dim[dA->num_dim - 1], dA->dim[0] * dA->stride[0] / dA->stride[dA->num_dim-3]);
//     return dC;
// }

//자 예를들어서 2x3x4x5 와 1x4x5 를 matmul하는 경우가 있다고 해보자. 그럴 경우 위의 matmul과는 다르게 2x3개의 행렬이서 matmul을 하는게 아니라
// 2x3을 1개의 행렬에 대해서 matrix-wise하는 것이다. 그럴 경우 conn_dim을 ....ㄱ ㅡㄴ데 굳이 conn_dim을 표시할 이유가 있을까. 
//2x3x4x5 와 3x4x5를 matmul한다 했을 때 일단 num_dim에서 차이가 나니까 그 다음에 행렬이 같은지를 확인을 해보는건 어떨까.
//어차피 matmul 함수에서도 각각의 dim이 같은지 다른지 확인을 하니까 어차피 해야할 작업이라는 것이다. 
//그러므로 먼저 num_dim 확인, 작은 num_dim을 기준으로 뒤에서부터 차이가 나는지 확인, 
//그다음 같으면 작은 num_dim을 기준으로 matrix로 flatten하면 
//작은 놈은 3차원, 큰놈은 4차원이 될 것이다. 2x3x4x5x6인데 3x4x5x6이라 하면 conn_dim=1이겠고, 
//큰놈 2x12x5x6, 작은 놈 12x5x6 일 것이다. 
//이것을 한번 4x3x16x16와 3x32x16에 한번 적용을 해보자. 
//예를들어 out_dim에 맞게 4x(in_channel)x16x16의 이미지를 4x(in_channel)x49x9로 만들고,
// flatten한 kernel (in_channel)x49x(out_channel)로 맞추어 했다고 치자. 기존 matmul함수와는 다르게
//4(batch_size)은 반복적으로, (in_channel)은 차원에 맞게 연산을 해주어야한다. 그러므로 이 작업은 타당하다. 
//1. num_dim이 맞는지 확인
//2. 작은 num_dim만큼의 차원들은 맞는지 확인
//3. 차원이 맞다면 flatten하고 (큰 dim 남는차원) x(작은 dim 차원) x(row)x(col) 꼴로 matmul을 해서 되게한다.

void matmul_(float *dA, float *dB, float *dC, float *dBias, int M, int N, int K, int numofMat, int big_dim_stride, int big_A_True) {
    dim3 dimGrid((N + tile_SIZE - 1) / tile_SIZE, (M + tile_SIZE - 1) / tile_SIZE, numofMat); //dim은 4x3x32x32를 matmul하는 경우 12가 들어가게 된다.
    dim3 dimBlock(tile_SIZE, tile_SIZE);
    tiledMM<<<dimGrid, dimBlock>>>(dA, dB, dC, dBias, M, N, K, big_dim_stride, big_A_True);
}

Tensor* matmul(Tensor*dC, Tensor *dA, Tensor *dB){
    if (dA->device_type != dB->device_type) {
        printf("two source Matrix is on different device. dA : %d, dB: %d\n", dA->device_type, dB->device_type);
        return NULL;
    }
    if (dC->device_type != dA->device_type) {
        printf("result matrix is on different device.dA : %d, dC: %d\n", dA->device_type, dC->device_type);
        return NULL;
    }

    if(dA->dim[dA->num_dim - 1] != dB->dim[dB->num_dim-2]){//dA와 dB의 row, col이 맞는지를 확인하는 작업.
        printf("number of column of dA and row of dB doesn't match\n");
        return NULL;
    }

    int big_A_True = 0;//이건 어느게 더 큰놈인지 판단을 하는 것. A가 크면 1
    
    Tensor* bigTensor, *smallTensor;//큰놈을 큰놈에, 작은놈을 작은놈에
    if(dA->num_dim > dB->num_dim){
        big_A_True = 1;
        bigTensor = dA;
        smallTensor = dB;
    }else{
        big_A_True = 0;
        bigTensor = dB;
        smallTensor = dA;
    }
    
    int dim_contrast = bigTensor->num_dim - smallTensor->num_dim;

    for(int i= smallTensor->num_dim - 3; i >= 0; i--){// bigTensor와 smallTensor의 차원을 서로 비교하는 것.
        if(bigTensor->dim[i + dim_contrast] != smallTensor-> dim[i]){
            printf("matrices have different dimension.\n");
            return NULL;
        }
    }

    if(dC->num_dim != bigTensor->num_dim){
        printf("matrices have different dimension.(num_dim is different)\n");
        return NULL;
    }

    for(int i=0; i < bigTensor->num_dim-3; i++){
        if(dA->dim[i] != dC->dim[i]){
            printf("dim %d of source and result Matrix is not the same.\n", i);
            return NULL;
        }
    }
    if(dA->dim[dA->num_dim - 2] != dC->dim[dA->num_dim - 2] || dC->dim[dC->num_dim -1] != dB->dim[dB->num_dim - 1]){
        printf("result matrix is in different dimension.\n");
        return NULL;
    }
    int big_dim_stride = bigTensor->stride[dim_contrast-1] / bigTensor->stride[bigTensor->num_dim-3];
    matmul_(dA->T, dB->T, dC->T, NULL, dA->dim[dA->num_dim - 2], dB->dim[dB->num_dim - 1], dA->dim[dA->num_dim - 1], bigTensor->dim[0] * bigTensor->stride[0] / bigTensor->stride[bigTensor->num_dim-3], big_dim_stride, big_A_True);
    return dC;

}

