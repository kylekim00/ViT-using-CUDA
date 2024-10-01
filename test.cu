#include<stdio.h>
#include<stdlib.h>
#include"./Easy_Tensor/easy_tensor.h"
#include<string.h>
Tensor* dummyTensor(Tensor *ten){
    for(int i=0; i < ten->dim[0] * ten->stride[0]; i++){
        ten->T[i] = i;
    }
    return ten;
}


Tensor* dummyTensor2(Tensor *ten){
    for(int i=0; i < ten->dim[0] * ten->stride[0]; i++){
        ten->T[i] = i%2;
    }
    return ten;
}
int main(){
    Tensor* A = dummyTensor(makeTensor("7 5", 0));
    Tensor* B = dummyTensor2(makeTensorbyShape(A,0));
    scalar_Tensor(A, '+', 3);
    scalar_Tensor(A, '-', 3);
    scalar_Tensor(A, '*', 3);
    scalar_Tensor(A, '+', 3);


    printTensor(A);


}