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
    Tensor** MHA_BLOCK[12];
    for(int i=0; i < 12; i++){
        MHA_BLOCK[i] = makeMHABlock(0);
    }

    MHA_BLOCK[0] = copyMHABlockfromFILE(MHA_BLOCK[0], "0_newblock.bin");



    Tensor**dMHA_block0 = copyMHABlock(makeMHABlock(1), MHA_BLOCK[0]);

    Tensor* O = makeTensor("4 196 768", 1);
    Tensor* O_proj  = makeTensor("4 196 768", 1);
    Tensor* dQKV = makeTensor("4 196 2304", 1);


    //dQKV check
    // Tensor* dd = copyTensor(makeTensorbyShape(dQKV, 0),dQKV);
    // freeTensor(printTensor(makeSubTensor(dd, "2 0 64", "8 8")));

    ///////ATTNTN////////


    dQKV = matmul_bias(dQKV, dInput, dMHA_block0[2], dMHA_block0[3], 0);//get QKV
    O = flashAttention_MHA(O, dQKV);//Flash Attention
    matmul_bias(O_proj, O, dMHA_block0[4], dMHA_block0[5], 0);//Projection

    // freeTensor(printTensor(makeSubTensor(copyTensor(makeTensorbyShape(O, 0), O), "2 188 0","8 16")));
    freeTensor(printTensor(makeSubTensor(copyTensor(makeTensorbyShape(O_proj, 0), O_proj), "2 188 760","8 8")));
    
    // infoTensor(dQKV);
    freeTensor(O);
    
    freeMHABlock(dMHA_block0);


    for(int i=0; i < 12; i++){
        freeMHABlock(MHA_BLOCK[i]);
    }
    freeTensor(input);
    freeTensor(dInput);
}