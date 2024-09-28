#include<stdio.h>
#include<stdlib.h>
#include"./Easy_Tensor/easy_tensor.h"
#include<string.h>
Tensor* dummyTensor(Tensor *ten){
    for(int i=0; i < ten->dim[0] * ten->stride[0]; i++){
        ten->T[i] = __FLT_MAX__/i;
    }
    return ten;
}
__global__ void CESoftmax_deriv_(float* deriv, float* O, float* label, int O_stride, int batch_size){
    int inx = blockDim.x * blockIdx.x + threadIdx.x;//each batch
    for(int i=0; i < O_stride;i++){
        if(i == label[inx]-1)
            deriv[O_stride * inx + i] = O[O_stride * inx + i] - 1;
        else
            deriv[O_stride * inx + i] = O[O_stride * inx + i];
    }
}
Tensor* CESoftmax_deriv(Tensor* d_der_O, Tensor*d_O, Tensor* d_Y){
    if(!d_der_O || !d_O || !d_Y){
        printf("CES: no Tensor.\n");
        return NULL;
    }
    if(d_O->dim[0] != d_Y->sizeTensor){
        printf("batch does not match.\n");
        return NULL;
    }
    if(d_der_O->num_dim != d_O->num_dim){
        printf("dimention does not match.\n");
        return NULL;
    }
    cudaSetDevice(d_der_O->device_type-1);
    CESoftmax_deriv_<<<(d_der_O->dim[d_der_O->num_dim - 2] + tile_SIZE - 1)/tile_SIZE, tile_SIZE>>>(d_der_O->T,d_O->T, d_Y->T, d_der_O->dim[d_der_O->num_dim - 1], d_der_O->dim[d_der_O->num_dim - 2]);
    return d_der_O;
}

Tensor* dummyTensor2(Tensor *ten){
    for(int i=0; i < ten->dim[0] * ten->stride[0]; i++){
        ten->T[i] = i+1;
    }
    return ten;
}
int main(){
    // Tensor *A = makeTensor("3 196 768", 0);
    // A = dummyTensor(A);
    // Tensor* B = dummyTensor(makeTensor("1 768 2304", 0));
    // // printTensor(ReLU_inline(B));
    // Tensor*bias = makeTensor("2304", 0);
    // for(int i=0; i < bias->sizeTensor; i++)
    //     bias->T[i] = i;
    
    // Tensor *dA = copyTensor(makeTensorbyShape(A, 1), A);
    // Tensor* dB = copyTensor(makeTensorbyShape(B, 1),B);

    // Tensor *dC = matmul_bias(makeTensor("3 196 2304",1), dA, dB, copyTensor(makeTensorbyShape(bias, 1), bias), 0);
    // // Tensor *dC = matmul(makeTensorbyShape(dB, 1), dA, dB);

    // printTensor(makeSubTensor(copyTensor(makeTensorbyShape(dC, 0), dC),"0 0 2296", "8 8"));
    ///////////////////////////////////////////////////////////////////////////////////////
    Tensor* A = dummyTensor(makeTensor("1 32", 0));
    Tensor* O = makeTensorbyShape(A, 0);
    Tensor* dA = copyTensor(makeTensorbyShape(A, 1), A);
    Tensor* dO = makeTensorbyShape(dA, 1);
    softMax(dO, dA);
    softMax(O, A);
    printTensor(O);

    printTensor(copyTensor(A, O));

    // Tensor* B = dummyTensor(makeTensor("16 40", 0));
    // Tensor* dB = copyTensor(makeTensorbyShape(B, 1), B);
    // Tensor* dAt = makeTensor("2 3 5 3", 1);
    // Tensor* At = makeTensor("2 2 5 3", 0);
    // Tensor* C = makeTensor("10 40", 0);
    // Tensor* dC = makeTensorbyShape(C, 1);

    
    // dA = scalar_Tensor(dA, '-', 0);
    // elementWise_Tensor(dA, dA, '*', dB); 

    // freeTensor(printTensor(copyTensor(At,copyTransposeTensor(dAt, dA))));


    // freeTensor(printTensor(copyTensor(makeTensor("16", 0), rowcolwise_sum(makeTensor("16", 1), dA, 1))));


    // Tensor* subA = makeSubTensor(dA, "0 0 0", "3 5 5");
    // printTensor(copyTensor(makeTensorbyShape(dB, 0),copyTransposeTensor(makeTensorbyShape(dB, 1),dB)));
    // printTensor(copyTensor(makeTensor("3 4 5 5", 0),matmul(makeTensor("3 4 5 5", 1), dA, dB));
    // Tensor* subA = makeSubTensor(A, "0 0 0 1", "3 1 2 2");
    
    // Tensor* dA = copyTensor(makeTensorbyShape(A, 1), A);
    // Tensor* dB = copyTensor(makeTensorbyShape(B, 1), B);
    // Tensor *dA_sub = makeSubTensor(dA, "0 0 0 1", "3 1 2 2");
    // int df[] = {0, 1, 2, 3};
    // printTensor(copyTensor(makeTensorbyShape(dA_sub, 0),copyReshapeTensor(makeTensorbyShape(dA_sub, 1), dA_sub, df)));
    // printTensor(B);
    // Tensor* dC = compMatMul(makeTensor("3, 3, 5, 2", 1), dB, dA_sub);
    // printTensor(copyTensor(makeTensorbyShape(dC, 0), dC));


    // Tensor*dAt = makeTensor("3, 2, 4, 4", 1);
    // Tensor*At = makeTensorbyShape(dAt, 0);
    // printf("dAt : %d", dAt->sizeTensor);
    // int df[] = {1, 2, 0, 3};
    // copyReshapeTensor(dAt, dA, df);
    // copyTensor(At, dAt);
    // printTensor(At);
    // freeTensor(At);

    // for(int i=0; i < 4; i++){
    //     df[i] = i;
    // }
    // Tensor* dsubA = makeSubTensor(dA, "0 0 0 1", "3 1 2 2");
    // printTensor(copyTensor(makeTensorbyShape(dsubA, 0),copyReshapeTensor(makeTensorbyShape(dsubA, 1), dsubA, df)));
    // infoTensor(dsubA);
    
    // Tensor* dC = compMatMul(makeTensor("3 3 2 2", 1), dsubA, copyTensor(makeTensorbyShape(B, 1), B));

    // printTensor(copyTensor(makeTensorbyShape(dC, 0), dC));

    


    // printTensor(A);
    // printTensor(subA);
    // freeTensor(subA);
}