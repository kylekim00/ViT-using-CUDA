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
    int dim[] = {5, 4, 2, 3};
    Tensor* A = makeTensor(dim, 4, 0);
    A = dummyTensor(A);
    int subDim[] = {4, 2, 3};
    int sp[] = {2, 0, 0, 0};
    Tensor* subA = makeSubTensor(A, sp, subDim, 3);
    int dim2[] = {3, 2, 4};
    Tensor* subAt = makeTensor(dim2, 3, 0);
    
    int resh[] = {2, 1, 0};
    copyReshapeTensor(subAt, subA, resh);

    printTensor(subAt);

    // Tensor* dA = copyTensor(makeTensorbyShape(A, 1), A);
    // Tensor* dAt = makeTensorbyShape(At, 1);

    // copyReshapeTensor(dAt, dA, resh);
    // copyTensor(At, dAt);
    // printTensor(At);

    // int sp[] = {0, 0, 2};//batch, numToken, Q
    // int sub_Dim[] = { 2, 2};
    // Tensor* subA = makeSubTensor(A, sp, sub_Dim, 2);
    // printTensor(subA);
    // freeTensor(subA);
    // sp[2] = 4;
    
    // subA = makelightcopysubTensor(A, sp, sub_Dim, 2);
    // printTensor(subA);
    // freeTensor(subA);

    freeTensor(A);
}