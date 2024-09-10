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
    int dim[] = {5, 1, 224, 224};
    Tensor* A = makeTensor(dim, 4, 0);
    A = dummyTensor(A);
    int subDim[] = {4, 224, 224};
    
    Tensor*B = makeTensor(subDim, 3, 0);
    B = dummyTensor(B);
    
    Tensor* dA = copyTensor(makeTensorbyShape(A, 1), A);
    Tensor* dB = copyTensor(makeTensorbyShape(B, 1), B);
    infoTensor(dA);
    infoTensor(dB);
    int resdim[] = {5, 4, 224, 224};
    Tensor* dC = makeTensor(resdim, 4, 1);
    compMatMul(dC, dA, dB);

    Tensor* C = copyTensor(makeTensorbyShape(dC, 0), dC);
    infoTensor(C);

    printf("%.02f", C->T[224 * 3 + 4]);
    

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