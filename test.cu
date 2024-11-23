#include<stdio.h>
#include<stdlib.h>
#include"./Easy_Tensor/easy_tensor.h"
#include"MHA.h"
#include<string.h>
Tensor* dummyTensor(Tensor *ten){
    for(int i=0; i < ten->dim[0] * ten->stride[0]; i++){
        ten->T[i] = 0.0001 * i;
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
    Tensor* A = dummyTensor(makeTensor("4 196 2304", 0));
    Tensor* dA = copyTensor(makeTensorbyShape(A, 1), A);

    Tensor* O = makeTensor("4 196 768", 0);

    Tensor* dO = makeTensorbyShape(O, 1);

    flashAttention_MHA(dO, dA);
    printTensor(copyTensor(O, dO));

    

    // input_cls = copyTensor(input_cls, dInput_cls);
    // infoTensor(dInput_cls);
    // infoTensor(dA);

    // printTensor(makeSubTensor(input_cls, "3 0 760", "8"));
    // printTensor(makeSubTensor(input_cls, "3 196 760", "8"));

    



    // Tensor* sub_dA = makeSubTensor(dA, "0 2 1", "3 4 3");

    // Tensor* C = makeTensor("3 4 7", 0);
    // printTensor(makeSubTensor(A, "0 2 1", "3 4 3"));
    // printTensor(B);
    // freeTensor(printTensor(copyTensor(C, matmul(makeTensorbyShape(C, 1), sub_dA, dB))));


    // scalar_Tensor(A, '+', 3);
    // scalar_Tensor(A, '-', 3);
    // scalar_Tensor(A, '*', 3);
    // scalar_Tensor(A, '+', 3);


    // printTensor(A);
    // Tensor** extra_weights = makeExtraWeights(0);
    // extra_weights = copyExtraWeightsfromFILE(extra_weights, "extra_weights.bin");
    // freeTensor(printTensor(makeSubTensor(extra_weights[4], "0", "8")));
    // Tensor** extra_weights_d = makeExtraWeights(1);
    // copyExtraWeights(extra_weights_d, extra_weights);
    
    // freeTensor(printTensor(makeSubTensor(copyExtraWeights(makeExtraWeights(0), extra_weights)[0], "0 0", "8 8")));
    // freeExtraWeights(extra_weights);



}