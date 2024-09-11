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
    Tensor* dA = copyTensor(makeTensorbyShape(A, 1), A);


    Tensor*dAt = makeTensor("3, 2, 4, 4", 1);
    Tensor*At = makeTensorbyShape(dAt, 0);
    printf("dAt : %d", dAt->sizeTensor);
    int df[] = {1, 2, 0, 3};
    copyReshapeTensor(dAt, dA, df);
    copyTensor(At, dAt);
    printTensor(At);
    freeTensor(At);

    for(int i=0; i < 4; i++){
        df[i] = i;
    }
    Tensor* dsubA = makeSubTensor(dA, "0 0 0 1", "3 1 2 2");
    printTensor(copyTensor(makeTensorbyShape(dsubA, 0),copyReshapeTensor(makeTensorbyShape(dsubA, 1), dsubA, df)));
    // infoTensor(dsubA);
    
    // Tensor* dC = compMatMul(makeTensor("3 3 2 2", 1), dsubA, copyTensor(makeTensorbyShape(B, 1), B));

    // printTensor(copyTensor(makeTensorbyShape(dC, 0), dC));

    


    // printTensor(A);
    // printTensor(subA);
    // freeTensor(subA);
}