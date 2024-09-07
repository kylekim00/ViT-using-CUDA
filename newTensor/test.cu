#include<stdio.h>
#include<stdlib.h>
#include"tensor_struct_2.h"
Tensor* dummyTensor(Tensor *ten){
    for(int i=0; i < ten->dim[0] * ten->stride[0]; i++){
        float dm = i;
        ten->T[i] = dm;
    }
    return ten;
}

int main(){
    int dim[] = {4, 3, 3, 2};
    Tensor* A = makeTensor(dim, 4, 0);
    // A->T[1] = 1;
    // A->T[2] = 2;
    A = dummyTensor(A);
    
    for(int i=0; i < A->num_dim; i++){
        printf("%d\t", A->stride[i]);
    }
    printf("\n");
    infoTensor(A);
    printTensor(A);
    Tensor* dA = copyTensor(makeTensorbyShape(A, 1), A);
    A = copyTensor(A, dA);
    infoTensor(dA);
    printTensor(A);
    freeTensor(dA);
    freeTensor(A);
}