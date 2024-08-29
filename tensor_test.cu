#include<stdio.h>
#include<stdlib.h>
#include "tensor_struct.h"

#include<iostream>
Tensor* dummyTensor(Tensor *ten){
    for(int i=0; i < ten->dim[0] * ten->stride[0]; i++){
        float dm = i;
        ten->T[i] = dm;
    }
    return ten;
}
int main(){
    int dim[] = {2,3, 32, 32};
    int dim2[] = {2, 3, 32, 4};
    Tensor *A = makeTensor(dim, sizeof(dim)/sizeof(int), 0);
    Tensor *B = makeTensor(dim2, sizeof(dim2)/sizeof(int), 0);

    Tensor *dA = makeTensor(dim, sizeof(dim)/sizeof(int), 1);
    Tensor *dB = makeTensor(dim2, sizeof(dim2)/sizeof(int), 1);

    Tensor *dC = makeTensor(dim2, sizeof(dim2)/sizeof(int), 1);
    A = dummyTensor(A);
    B = dummyTensor(B);
    dA = copyTensor(dA, A);
    dB = copyTensor(dB, B);
    matmul(dC, dA, dB);
    B = copyTensor(B, dC);
    // A = copyTensor(A, B);
    // for(int i=0; i < sizeof(dim)/sizeof(int); i++){
    //     printf("%d " , A->dim[i]);
    // }
    // printf("\n");
    // for(int i=0; i < sizeof(dim)/sizeof(int); i++){
    //     printf("%d " , A->stride[i]);
    // }
    printf("\n%d\n", B->stride[0] * B->dim[0]);
    // printf("%d\n", sizeof(A->T)/sizeof(float));
    printTensor(B);
    freeTensor(A);
}