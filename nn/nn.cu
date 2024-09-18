#include<stdio.h>
#include<stdlib.h>
#include "./../easy_tensor.h"

Tensor* copyTensorfromFILE(Tensor* dst, char* file_name);


__global__ void softmax_(float* T, int batch, int label_num){//inx는 batch 크기이다. 
    int inx = blockIdx.x * blockDim.x + threadIdx.x;
    if(inx < batch){
        float max = -__FLT_MAX__;
        float sum = 0;
        for(int i=0; i < label_num; i++){
           max = T[inx * label_num + i];
        }
        for(int i=0; i < label_num; i++){
            sum += expf(T[inx * label_num + i]-max);
        }
        for(int i=0; i < label_num; i++){
            T[inx * label_num + i] = expf(T[inx * label_num + i]-max)/sum;
        }
    }
    __syncthreads();
}

Tensor* softmax(Tensor* ten){
    if(ten->device_type){
        cudaSetDevice(ten->device_type - 1);
        softmax_<<<(ten->sizeTensor + tile_SIZE - 1)/tile_SIZE, tile_SIZE>>>(ten->T, ten->dim[0], ten->dim[1]);
    }else{
        for(int i=0; i < ten->dim[0]; i++){
            float max =  -__FLT_MAX__;
            float sum = 0;
            for(int j=0; j < ten->dim[1]; j++){
                if(ten->T[i * ten->stride[0] + j] > max)
                    max = ten->T[i * ten->stride[0]];
            }
            for(int j=0; j < ten->dim[1]; j++){
                sum += expf(ten->T[i * ten->stride[0] + j]-max);
            }
            for(int j=0; j < ten->dim[1]; j++){
                ten->T[i * ten->stride[0] + j] = expf(ten->T[i * ten->stride[0] + j]-max)/sum;
            }
        }

    }
    return ten;
}

Tensor* dSoftMax_cross_entropy(Tensor* O, Tensor* Y){
    
}

// Tensor* pred_(Tensor* pred, Tensor* O){

// }

int main(){
    Tensor* input_batch = makeTensor("4 784", 0);
    
    Tensor* d_input_batch = makeTensorbyShape(input_batch, 1);

    infoTensor(input_batch);
    Tensor* W[4];
    W[0] = makeTensor("784 50", 0);
    W[1] = makeTensor("50 30", 0)
    W[2] = makeTensor("30 40", 0);
    W[3] = makeTensor("40 10", 0);
    for(int i=0; i < sizeof(W)/sizeof(Tensor*); i++){
        infoTensor(W[i]);
        //put init weight in...
    }
    
    Tensor* dW[4];//loaded weight to device
    for(int i=0; i < sizeof(dW)/sizeof(Tensor*); i++){
        dW[i] = copyTensor(makeTensorbyShape(W[i], 1), W[i]);
    }

    Tensor* A[4];
    A[0] = makeTensor("4 50", 1);
    A[1] = makeTensor("4 30", 1);
    A[2] = makeTensor("4 40", 1);
    A[3] = makeTensor("4 10", 1);
    Tensor* O = makeTensorbyShape(A[3], 0);
    //forward pass
    A[0] = matmul(A[0], d_input_batch, dW[0]);
    for(int i=0; i < 3; i++){
        A[i] = ReLU_inline(A[i]);
        A[i+1] = matmul(A[i+1], A[i], dW[i+1]);
    }
    // O = printTensor(copyTensor(O, A[3]));
    A[3] = softmax(A[3]);

    Tensor* der_A[4];
    der_A[0] = makeTensor("4 50", 1);
    der_A[0] = makeTensor("4 30", 1);
    der_A[0] = makeTensor("4 40", 1);
    der_A[0] = makeTensor("4 10", 1);
    
    Tensor* der_W[4];
    der_W[0] = makeTensor()





    for(int i=0; i< sizeof(W)/ sizeof(Tensor*); i++){
        freeTensor(A[i]);
        freeTensor(dW[i]);
        freeTensor(W[i]);
    }
    freeTensor(d_input_batch);
    freeTensor(input_batch);
    
    
    
    
}