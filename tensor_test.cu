#include<stdio.h>
#include<stdlib.h>
#include "tensor_struct.h"

Tensor* dummyTensor(Tensor *ten){
    for(int i=0; i < ten->dim[0] * ten->stride[0]; i++){
        float dm = i;
        ten->T[i] = dm;
    }
    return ten;
}
int main(){
    int dim[] = {2, 3, 4, 5};
    int dim2[] = {3, 5, 4};
    Tensor *A = makeTensor(dim, sizeof(dim)/sizeof(int), 0);
    Tensor *At = makeTensor(dim2, sizeof(dim2)/sizeof(int), 0);
    A = dummyTensor(A);
    At = dummyTensor(At);
    // int reshape[] = {0, 1, 3, 2};
    // reshapeTensor(At, A, reshape);//이거 근데 copy랑 다를게 없지 않나
    printTensor(A);
    printTensor(At);
    Tensor *dA = copyTensor(makeTensorbyShape(A, 1), A);
    Tensor* dAt = makeTensorbyShape(At, 1);
    copyTensor(dAt, At);
    int dim3[] = {2, 3, 4, 4};
    Tensor* dC = makeTensor(dim3, sizeof(dim3)/sizeof(int), 1);
    Tensor* C = makeTensorbyShape(dC, 0);

    matmul_matwise(dC, dA, dAt);

    copyTensor(C, dC);
    printTensor(C);

    freeTensor(A);
    freeTensor(At);
    
}