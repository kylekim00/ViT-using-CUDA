#include<stdio.h>
#include<stdlib.h>
#include"easy_tensor.h"
#include<string.h>
Tensor* dummyTensor(Tensor *ten){
    for(int i=0; i < ten->dim[0] * ten->stride[0]; i++){
        float dm = i;
        ten->T[i] = dm;
    }
    return ten;
}


int main(){
    Tensor *A = makeTensor("4 3 2 4", 0);
    A = dummyTensor(A);
    printTensor(A);
    Tensor* B = makeTensor("3, 2, 5", 0);
    B = dummyTensor(B);
    // Tensor* subA = makeSubTensor(A, "0 0 0 1", "3 1 2 2");
    Tensor* dA = makeTensorbyShape(A, 1);

    Tensor* dsubA = makeSubTensor(dA, "0 0 0 1", "3 1 2 2");
    // infoTensor(dsubA);
    int df[] = {0, 1, 2, 3};
    printTensor(copyTensor(makeTensorbyShape(dsubA, 0),copyReshapeTensor(makeTensorbyShape(dsubA, 1), dsubA, df)));
    infoTensor(dsubA);
    
    // Tensor* dC = compMatMul(makeTensor("3 3 2 2", 1), dsubA, copyTensor(makeTensorbyShape(B, 1), B));

    // printTensor(copyTensor(makeTensorbyShape(dC, 0), dC));

    


    // printTensor(A);
    // printTensor(subA);
    // freeTensor(subA);
}