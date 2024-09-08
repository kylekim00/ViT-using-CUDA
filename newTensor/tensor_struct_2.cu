#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include "tensor_struct_2.h"
//나중에는 나눠서 각각의 gpu 안에 넣어야하기 때문에 생각을 해보면 인덱스 값에 따라 값을 copy 해주는 것도 있으면 좋을 것 같다.
//기존 텐서와 다른점. 

//1. dim과 stride가 다를 때를 고려한다. 아울러 총 크기를 그냥 저장해버린다. 이는 makelightcopysubTensor() 함수를 만들기 위함이다. 마치 커서의 드래그와 같은 역할을 해줄 것이다.(해결)
//2. num_dim이 1, 2 일때도 작동이 되도록한다.
//3. device 위에 올라가 있을 경우 dim과 stride도 같이 device에 올려준다. [5, 1, 3, 3, 4] X [4, 1, 4, 2] 와 같은 복잡한 텐서도 행렬곱이 가능하게 하기 위함이다. 

Tensor *makeTensor(int *dim, int num_dim, int device_type){
    if(!dim){                       //if There is no dim inside
        return NULL;
    }
    if(num_dim < 0){                //if there is not an appropriate num_dim
        return NULL;
    }
    int device;
    cudaGetDeviceCount(&device);
    if(device_type > device){       //count device and check the boundary
        printf("DEVICE NUM %d NOT AVAILABLE\n", device_type);
        return NULL;
    }

    int sizeTensor;                 //check the size of whole Tensor

    Tensor* ten = (Tensor*)malloc(sizeof(Tensor));      //give tensor a space for host
    ten->dim = (int*)malloc(2 * num_dim * sizeof(int)); //give dim and stride a spcae for host
    ten->stride = ten->dim+num_dim;                    //this approach might be effective when sending to GPU later.

    ten->num_dim = num_dim;
    ten->device_type = device_type;

    sizeTensor = 1;
    for(int i= num_dim - 1; i >= 0; i--){
        ten->dim[i] = dim[i];
        ten->stride[i] = sizeTensor;
        sizeTensor *= dim[i];
    }

    ten->sizeTensor = sizeTensor;

    if(!device_type){
        ten->T = (float*)malloc(sizeTensor * sizeof(float));
        ten->d_dim_stride = NULL;
    }else{
        cudaSetDevice(device_type-1);
        cudaMalloc(&ten->T, sizeTensor * sizeof(float));
        cudaMalloc(&ten->d_dim_stride, 2 * num_dim * sizeof(int));
        cudaMemcpy(ten->d_dim_stride, ten->dim, 2 * num_dim * sizeof(int), cudaMemcpyHostToDevice);
    }
    ten->isSub = 0;
    return ten;
}


Tensor* makeTensorbyShape(Tensor* src, int device_type){
    if(!src){
        printf("SouRCe is vacant.\n");
        return NULL;
    }
    if(src->isSub){
        printf("Source is SubTensor.\n");
        return NULL;
    }
    return makeTensor(src->dim, src->num_dim, device_type);
}


Tensor* makeSubTensor(Tensor* src, int* start_point, int* dim, int num_dim){
    if(src->isSub){
        printf("Cant light copy subTensor\n");
        return NULL;
    }
    if(src->num_dim < num_dim){
        printf("SouRCe num_dim not that big\n");
        return NULL;
    }

    int cont = src->num_dim - num_dim;
    float* sp = src->T;


    for(int i=0; i < src->num_dim; i++){                            //This is where you set the starting point
        if(src->dim[i] <= start_point[i]){
            printf("starting point invalid\n");
            return NULL;
        }
        sp += start_point[i] * src->stride[i];
    }

    for(int i=0; i < num_dim; i++){                                 //This is where tou check the size of the dim 
        if(src->dim[i + cont] < start_point[i+cont] + dim[i]){
            printf("SouRCe not that big.\n");
            return NULL;
        }
    }

    Tensor* subTensor = (Tensor*)malloc(sizeof(Tensor));            //Tensor malloc

    subTensor->isSub = 1;
    subTensor->device_type = src->device_type;                      //device_type same as src
    subTensor->num_dim = num_dim;                                   //num_dim

    subTensor->dim = (int*)malloc(2 * sizeof(int)* num_dim);        //dim stride malloc
    subTensor->stride = subTensor->dim + num_dim;

    subTensor->sizeTensor = 1;
    for(int i=0; i < num_dim; i++){                             
        subTensor->dim[i] = dim[i];
        subTensor-> stride[i] = src->stride[i+cont];            //copy Stride
        subTensor->sizeTensor *= dim[i];
    }

    subTensor->T = sp;

    if(src->device_type){
        cudaSetDevice(src->device_type-1);
        cudaMalloc(&subTensor->d_dim_stride, 2 * num_dim * sizeof(int));
        cudaMemcpy(subTensor->d_dim_stride, subTensor->dim, 2 * num_dim * sizeof(int), cudaMemcpyHostToDevice);
    }else{
        subTensor->d_dim_stride = NULL;
    }

    return subTensor;
}



//================================================FREEEEEEEEEEEE===============================================================

void freeSubTensor(Tensor* subTen){
    if(!subTen->isSub){
        printf("This is Not a SubTensor.\n");
        return;
    }
    if(subTen==NULL){
        printf("NO TENSOR IN POINTER.\n");
        return;
    }else{
        if(subTen->device_type){
            cudaSetDevice(subTen->device_type - 1);
            cudaError_t err = cudaFree(subTen->d_dim_stride);
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
            }
        }
        free(subTen->dim);         //didn't malloc ten->stride from the first place :P
        free(subTen);
    }
}
void freeTensor(Tensor *ten){
    if(ten==NULL){
        printf("NO TENSOR IN POINTER.\n");
        return;
    }
    if(ten->isSub){
        freeSubTensor(ten);
        return;
    }else{
        if(ten->device_type){
            cudaSetDevice(ten->device_type - 1);
            cudaError_t err = cudaFree(ten->T);
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
            }
            err = cudaFree(ten->d_dim_stride);
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
            }
        }else{
            free(ten->T);
        }
    }
    free(ten->dim);         //didn't malloc ten->stride from the first place :P
    free(ten);
}

//=================================print================================

void infoTensor(Tensor *ten){
    printf("\n=========Tensor===========\n");
    if(ten->isSub){
        printf("===SUBTENSOR===\n");
    }
    printf("DIMENTION : [");
    for(int i=0; i < ten->num_dim-1; i++){
        printf("%d ", ten->dim[i]);
    }
    printf("%d]\n", ten->dim[ten->num_dim - 1]);
    printf("STRIDE    : [");
    for(int i=0; i < ten->num_dim-1; i++){
        printf("%d ", ten->stride[i]);
    }
    printf("%d]\n", ten->stride[ten->num_dim - 1]);
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
    infoTensor(ten);
    //==============if ten->num_dim < 3========================
    if(ten->num_dim == 1){
        printf("[ %d ]\n", ten->dim[0]);
        for(int i=0; i < ten->dim[0]; i+=ten->stride[0]){
            printf("%.02f\t", ten->T[i]);
        }
        printf("\n");
        return;
    }
    if(ten->num_dim == 2){
        printf("[ %d %d ]\n", ten->dim[0], ten->dim[1]);
        for(int i=0; i < ten->dim[0]*ten->stride[0]; i+=ten->stride[0]){
            for(int j=0; j < ten->dim[1]*ten->stride[1]; j+=ten->stride[1]){
                printf("%.02f\t", ten->T[i + j]);
                // printf("%d\t", ten->stride[0]* i + j);
            }
            printf("\n");
        }
        printf("\n");
        return;
    }
    //=================else====================================
    printf("=\n");

    int* tmp_Inx = (int*)malloc(sizeof(int) * (ten->num_dim - 2));
    for(int i=0; i < ten->num_dim - 2; i++){
        tmp_Inx[i] = 0;
    }
    int inx;
    while(tmp_Inx[0] < ten->dim[0]){
        inx = 0;
        printf("[ ");
        for(int i=0; i < ten->num_dim-2;i++){
            printf("%d ", tmp_Inx[i]);
            inx += tmp_Inx[i] * ten->stride[i];
        }
        printf("- - ]\n");

        for(int i=0; i < ten->dim[ten->num_dim-2]*ten->stride[ten->num_dim-2]; i+=ten->stride[ten->num_dim-2]){
            for(int j=0; j < ten->dim[ten->num_dim-1]*ten->stride[ten->num_dim-1]; j+=ten->stride[ten->num_dim-1]){
                printf("%.02f\t", ten->T[inx + i + j]);
                // printf("%d\t", ten->stride[0]* i + j);
            }
            printf("\n");
        }

        tmp_Inx[ten->num_dim - 3]++;
        for(int i = ten->num_dim - 3; i > 0; i--){
            if(tmp_Inx[i] >= ten->dim[i]){
                tmp_Inx[i-1]++;
                tmp_Inx[i] = 0;
            }
        }
    }
    printf("=\n");
    free(tmp_Inx);
}

Tensor* copyTensor(Tensor *dst, Tensor *src){
    if(!dst || !src){
        printf("No dst or src\n");
        return NULL;
    }
    if(dst->isSub||src->isSub){
        printf("dst or src is subTensor.");
        return NULL;
    }
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


__global__ void reshape_(float* dst, float* src, int* dst_dim_stride, int* src_dim_stride, int* reshape, int num_dim, int sizeTensor){
    src_dim_stride += num_dim;
    dst_dim_stride += num_dim;
    int new_inx = blockDim.x * blockIdx.x + threadIdx.x;
    int new_tmp = new_inx;
    int inx = 0;
    for(int i=0; i < num_dim; i++){
        inx += new_tmp / dst_dim_stride[i] * src_dim_stride[reshape[i]];
        new_tmp %= dst_dim_stride[i];
    }
    if(inx < sizeTensor)
        dst[new_inx] = src[inx];


}


Tensor* copyReshapeTensor(Tensor* dst, Tensor* src, int* reshape){
    if(dst->isSub){
        printf("dst can't be subMatrix.\n");
    }
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

    if(src->device_type){//GPU
        cudaSetDevice(src->device_type - 1);
        int* d_tmp_reshape;
        cudaMalloc(&d_tmp_reshape, sizeof(int) * dst->num_dim);
        cudaMemcpy(d_tmp_reshape, reshape, sizeof(int) * dst->num_dim, cudaMemcpyHostToDevice);
        
        int s_tile_SIZE = tile_SIZE * tile_SIZE;
        
        reshape_<<< (dst->sizeTensor + s_tile_SIZE - 1)/s_tile_SIZE, s_tile_SIZE >>>(dst->T, src->T, dst->d_dim_stride, src->d_dim_stride, d_tmp_reshape, src->num_dim, dst->sizeTensor);
        cudaFree(d_tmp_reshape);

    }else{//CPU
        int newInx, inx;
        for(int new_inx =0; new_inx < dst->sizeTensor; new_inx++){
            newInx = new_inx;
            inx = 0;
            for(int i=0; i < src->num_dim; i++){
                inx += newInx / dst->stride[i] * src->stride[reshape[i]];
                newInx = newInx % dst->stride[i];
            }
            dst->T[new_inx] = src->T[inx];
        }
    }

    return dst;
}


// __global__ void tiledMM(float *A, float *B, float *C, float *bias, int M, int N, int K, int big_dim_stride, int big_A_True) {
//     //blockDim.z, blockIdx.z 이게 매트릭스의 수//z thread를 늘린다고 문제가 해결되지 않는다.
//     int matIdx_A, matIdx_B;
//     if(big_A_True){
//         matIdx_A = blockDim.z * blockIdx.z;
//         matIdx_B = matIdx_A % big_dim_stride;
//     }else{
//         matIdx_B = blockDim.z * blockDim.z;
//         matIdx_A = matIdx_B % big_dim_stride;
//     }
    
    
//     int row = blockDim.y * blockIdx.y + threadIdx.y;
//     int col = blockDim.x * blockIdx.x + threadIdx.x;
//     __shared__ float s_a[tile_SIZE][tile_SIZE];
//     __shared__ float s_b[tile_SIZE][tile_SIZE];

//     float tmp = 0.0f;

//     for (int i = 0; i < (K + tile_SIZE - 1) / tile_SIZE; i++) {
//         if (row < M && (i * tile_SIZE + threadIdx.x) < K)
//         //A와 B에 데이터를 넣을 때 matIdx에 차이를 두어야 한다.
//         //예를 들자면 matIdxA = blockDim.z * blockIdx.z / big_Dim_stride; 이러면 반복이 되니까.
//             s_a[threadIdx.y][threadIdx.x] = A[matIdx_A * (K*M) + row * K + (i * tile_SIZE + threadIdx.x)];
//         else
//             s_a[threadIdx.y][threadIdx.x] = 0.0f;

//         if (col < N && (i * tile_SIZE + threadIdx.y) < K)
//             s_b[threadIdx.y][threadIdx.x] = B[matIdx_B * (K*N) +(i * tile_SIZE + threadIdx.y) * N + col];
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
//             tmp = tmp + bias[col];
//         if(big_A_True)
//             C[matIdx_A *(M*N) + row * N + col] = tmp;
//         else
//             C[matIdx_B *(M*N) + row * N + col] = tmp;
//     }
// }

// void matmul_matwise_(float *dA, float *dB, float *dC, float *dBias, int M, int N, int K, int numofMat, int big_dim_stride, int big_A_True) {
//     dim3 dimGrid((N + tile_SIZE - 1) / tile_SIZE, (M + tile_SIZE - 1) / tile_SIZE, numofMat); //dim은 4x3x32x32를 matmul하는 경우 12가 들어가게 된다.
//     dim3 dimBlock(tile_SIZE, tile_SIZE);
//     tiledMM<<<dimGrid, dimBlock>>>(dA, dB, dC, dBias, M, N, K, big_dim_stride, big_A_True);
// }

// Tensor* matmul_matwise(Tensor*dC, Tensor *dA, Tensor *dB){
//     if (dA->device_type != dB->device_type) {
//         printf("two source Matrix is on different device. dA : %d, dB: %d\n", dA->device_type, dB->device_type);
//         return NULL;
//     }
//     if (dC->device_type != dA->device_type) {
//         printf("result matrix is on different device.dA : %d, dC: %d\n", dA->device_type, dC->device_type);
//         return NULL;
//     }

//     if(dA->dim[dA->num_dim - 1] != dB->dim[dB->num_dim-2]){//dA와 dB의 row, col이 맞는지를 확인하는 작업.
//         printf("number of column of dA and row of dB doesn't match\n");
//         return NULL;
//     }

//     int big_A_True = 0;//이건 어느게 더 큰놈인지 판단을 하는 것. A가 크면 1
    
//     Tensor* bigTensor, *smallTensor;//큰놈을 큰놈에, 작은놈을 작은놈에
//     if(dA->num_dim > dB->num_dim){
//         big_A_True = 1;
//         bigTensor = dA;
//         smallTensor = dB;
//     }else{
//         big_A_True = 0;
//         bigTensor = dB;
//         smallTensor = dA;
//     }
    
//     int dim_contrast = bigTensor->num_dim - smallTensor->num_dim;

//     for(int i= smallTensor->num_dim - 3; i >= 0; i--){// bigTensor와 smallTensor의 차원을 서로 비교하는 것.
//         if(bigTensor->dim[i + dim_contrast] != smallTensor-> dim[i]){
//             printf("matrices have different dimension.\n");
//             return NULL;
//         }
//     }

//     if(dC->num_dim != bigTensor->num_dim){
//         printf("matrices have different dimension.(num_dim is different)\n");
//         return NULL;
//     }

//     for(int i=0; i < bigTensor->num_dim-3; i++){
//         if(dA->dim[i] != dC->dim[i]){
//             printf("dim %d of source and result Matrix is not the same.\n", i);
//             return NULL;
//         }
//     }
//     if(dA->dim[dA->num_dim - 2] != dC->dim[dA->num_dim - 2] || dC->dim[dC->num_dim -1] != dB->dim[dB->num_dim - 1]){
//         printf("result matrix is in different dimension.\n");
//         return NULL;
//     }
//     int big_dim_stride = bigTensor->stride[dim_contrast-1] / bigTensor->stride[bigTensor->num_dim-3];
//     matmul_matwise_(dA->T, dB->T, dC->T, NULL, dA->dim[dA->num_dim - 2], dB->dim[dB->num_dim - 1], dA->dim[dA->num_dim - 1], bigTensor->dim[0] * bigTensor->stride[0] / bigTensor->stride[bigTensor->num_dim-3], big_dim_stride, big_A_True);
//     return dC;
// }







