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
    Tensor* attn_Residual = makeTensorbyShape(dInput, dInput->device_type);//이거는 Attention블럭 들어가기 전에 있어야한다. residual 전달해야됌
    Tensor* dQKV = makeTensor("4 196 2304", 1);
    Tensor* attn_mlp = makeTensor("4 196 3072", 1);



    //dQKV check
    // Tensor* dd = copyTensor(makeTensorbyShape(dQKV, 0),dQKV);
    // freeTensor(printTensor(makeSubTensor(dd, "2 0 64", "8 8")));

    
    ///////ATTNTN////////
    //residual store
    copyTensor(attn_Residual, dInput);
    printf("=input=\n");
    freeTensor(printTensor(makeSubTensor(copyTensor(makeTensorbyShape(dInput, 0), dInput), "0 0 0","8 8")));
    //normalize1
    normalize(dInput, dInput);
    elementWise_Tensor(dInput, dInput, '*', dMHA_block0[0]);//여기의 dMHA_BLOCK은 broadcasting 을 해야한다.
    printf("=norm=\n");
    freeTensor(printTensor(makeSubTensor(copyTensor(makeTensorbyShape(dInput, 0), dInput), "0 0 0","8 8")));
    elementWise_Tensor(dInput, dInput, '+', dMHA_block0[1]);

    freeTensor(printTensor(makeSubTensor(copyTensor(makeTensorbyShape(dInput, 0), dInput), "0 0 0","8 8")));
    //QKV
    dQKV = matmul_bias(dQKV, dInput, dMHA_block0[2], dMHA_block0[3], 0);//get QKV

    //flashAttention
    O = flashAttention_MHA(O, dQKV);//Flash Attention
    
    //projection
    O_proj = matmul_bias(O_proj, O, dMHA_block0[4], dMHA_block0[5], 0);//Projection
    freeTensor(printTensor(makeSubTensor(copyTensor(makeTensorbyShape(O_proj, 0), O_proj), "0 0 0","8 8")));
    
    //residual 1
    O_proj = elementWise_Tensor(O_proj, O_proj, '+', attn_Residual);
    copyTensor(attn_Residual, O_proj);

    //normalize2
    normalize(O_proj, O_proj);
    elementWise_Tensor(O_proj, O_proj, '*', dMHA_block0[6]);
    elementWise_Tensor(O_proj, O_proj, '+', dMHA_block0[7]);

    //MLP layer
    attn_mlp = matmul_bias(attn_mlp, O_proj, dMHA_block0[8], dMHA_block0[9], 0);
    attn_mlp = gelu_Tensor(attn_mlp);
    O_proj = matmul_bias(O_proj, attn_mlp,dMHA_block0[10], dMHA_block0[11], 0);

    //residual 2
    O_proj = elementWise_Tensor(O_proj, O_proj, '+', attn_Residual);
    infoTensor(O_proj);
    freeTensor(printTensor(makeSubTensor(copyTensor(makeTensorbyShape(O_proj, 0), O_proj), "2 188 760","8 8")));
    // freeTensor(printTensor(makeSubTensor(copyTensor(makeTensorbyShape(O, 0), O), "2 188 0","8 16")));
    // infoTensor(dQKV);
    freeTensor(O);
    
    freeMHABlock(dMHA_block0);


    for(int i=0; i < 12; i++){
        freeMHABlock(MHA_BLOCK[i]);
    }
    freeTensor(input);
    freeTensor(dInput);
}