#include<stdio.h>
#include<stdlib.h>
#include "easy_tensor.h"

Tensor* copyTensorfromFILE(Tensor* dst, char* file_name);

int main(){
    Tensor* input_batch = makeTensor("4 1 784", 0);
    infoTensor(input_batch);
    Tensor* W[4];
    W[0] = makeTensor("784 50", 0);
    W[1] = makeTensor("50 30", 0);
    W[2] = makeTensor("30 40", 0);
    W[3] = makeTensor("40 10", 0);
    for(int i=0; i < sizeof(W)/sizeof(Tensor*); i++){
        infoTensor(W[i]);
    }
    
    Tensor* dW[4];//loaded weight to device
    for(int i=0; i < sizeof(dW)/sizeof(Tensor*); i++){
        dW[i] = copyTensor(makeTensorbyShape(W[i], 1), W[i]);
    }

    
    
}