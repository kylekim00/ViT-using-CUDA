#include<stdio.h>
#include<stdlib.h>
#include<float.h>
#include <curand.h>
#include <curand_kernel.h>
#include "./Easy_Tensor/easy_tensor.h"
#include "MHA.h"
#include<string.h>


int main(){
    char input_dim[] = "4 196 768";
    // char QKV_dim[] = "3 196 2304";
    //pretrained weight initialization
    

    //dummy input
    Tensor* input = makeTensor(input_dim, 0);
    input = copyTensorfromFILE(input, "dummy_input_4_196_768.bin");
    printf("=======input=======\n");
    // freeTensor(printTensor(makeSubTensor(input, "0 0 0", "8 8")));

    //input to device
    Tensor*dInput = copyTensor(makeTensorbyShape(input, 1), input);


    //MHA_block0 FILE copy
    Tensor **MHA_block0 = makeMHABlock(0);

    MHA_block0 = copyMHABlockfromFILE(MHA_block0, "block_0.bin");

    Tensor**dMHA_block0 = copyMHABlock(makeMHABlock(1), MHA_block0);

    Tensor* dQKV = matmul_bias(makeTensor("4 196 2304", 1),dInput, dMHA_block0[0], dMHA_block0[1], 0);



    //dQKV check
    // Tensor* dd = copyTensor(makeTensorbyShape(dQKV, 0),dQKV);
    // freeTensor(printTensor(makeSubTensor(dd, "2 0 64", "8 8")));

    ///////ATTNTN////////

    Tensor* O = makeTensor("4 196 768", 1);
    
    flashAttention_MHA(O, dQKV);

    // freeTensor(printTensor(makeSubTensor(copyTensor(makeTensorbyShape(O, 0), O), "2 188 0","8 16")));
    freeTensor(printTensor(makeSubTensor(copyTensor(makeTensorbyShape(O, 0), O), "1 0 0","8 16")));
    
    // infoTensor(dQKV);
    freeTensor(O);
    freeMHABlock(MHA_block0);
    freeMHABlock(dMHA_block0);
    freeTensor(input);
}