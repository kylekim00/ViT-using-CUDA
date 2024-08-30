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
    int dim[] = {2, 3, 4, 5};
    int dim2[] = {3, 5, 4, 2};
    Tensor *A = makeTensor(dim, sizeof(dim)/sizeof(int), 0);
    Tensor *At = makeTensor(dim2, sizeof(dim)/sizeof(int), 0);
    A = dummyTensor(A);
    
    int reshape[] = {1, 3, 2, 0};
    reshapeTensor(At, A, reshape);
    
    printTensor(At);
    // B = dummyTensor(B);
    // dA = copyTensor(dA, A);
    // dB = copyTensor(dB, B);
    // matmul(dC, dA, dB);
    // B = copyTensor(B, dC);
    // A = copyTensor(A, B);
    // for(int i=0; i < sizeof(dim)/sizeof(int); i++){
    //     printf("%d " , A->dim[i]);
    // }
    // printf("\n");
    // for(int i=0; i < sizeof(dim)/sizeof(int); i++){
    //     printf("%d " , A->stride[i]);
    // }
    // printf("\n%d\n", B->stride[0] * B->dim[0]);
    // printf("%d\n", sizeof(A->T)/sizeof(float));
    // printTensor(B);
    freeTensor(A);
    freeTensor(At);
    // freeTensor(dA);
    // freeTensor(dB);
    // freeTensor(dC);
}