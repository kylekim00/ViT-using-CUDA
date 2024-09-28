#include<stdio.h>
#include<stdlib.h>
#include"easy_tensor.h"
Tensor* dummyTensor(Tensor *ten){
    for(int i=0; i < ten->dim[0] * ten->stride[0]; i++){
        ten->T[i] = 1;
    }
    return ten;
}
int main(){
    Tensor* tmp = dummyTensor(makeTensor("3 4", 0));
    Tensor* dTmp = copyTensor(makeTensorbyShape(tmp, 1), tmp);

}

__global__ void sum_(float* dst, int len){
    
}
Tensor* sum(Tensor* ten){

}