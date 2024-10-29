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
    
    //dummy input
    Tensor* input = makeTensor(input_dim, 0);
    input = copyTensorfromFILE(input, "dummy_input_4_196_768.bin");
    // freeTensor(printTensor(makeSubTensor(input, "0 0 0", "8 8")));

    //input to device
    Tensor*dInput = copyTensor(makeTensorbyShape(input, 1), input);


    //pretrained weight initialization
    //MHA_block0 FILE copy
    Tensor** MHA_BLOCK[12];

    //EXTRA_weights copy
    Tensor** EXTRA_weights = makeExtraWeights(0);
    copyExtraWeightsfromFILE(EXTRA_weights, "extra_weights.bin");
    Tensor** EXTRA_weights_d = copyExtraWeights(makeExtraWeights(1), EXTRA_weights);

    


    for(int i=0; i < 12; i++){
        MHA_BLOCK[i] = makeMHABlock(0);
    }
    copyMHABlockfromFILE(MHA_BLOCK[0], "0_newblock.bin");
    copyMHABlockfromFILE(MHA_BLOCK[1], "1_newblock.bin");
    copyMHABlockfromFILE(MHA_BLOCK[2], "2_newblock.bin");
    copyMHABlockfromFILE(MHA_BLOCK[3], "3_newblock.bin");
    copyMHABlockfromFILE(MHA_BLOCK[4], "4_newblock.bin");
    copyMHABlockfromFILE(MHA_BLOCK[5], "5_newblock.bin");
    copyMHABlockfromFILE(MHA_BLOCK[6], "6_newblock.bin");
    copyMHABlockfromFILE(MHA_BLOCK[7], "7_newblock.bin");
    copyMHABlockfromFILE(MHA_BLOCK[8], "8_newblock.bin");
    copyMHABlockfromFILE(MHA_BLOCK[9], "9_newblock.bin");
    copyMHABlockfromFILE(MHA_BLOCK[10], "10_newblock.bin");
    copyMHABlockfromFILE(MHA_BLOCK[11], "11_newblock.bin");
    
    
    Tensor**dMHA_block = makeMHABlock(1);

    Tensor* O = makeTensor("4 196 768", 1);




    //Attention tmp tensors.
    int ATTN_layer_num = 12;
    Tensor* O_proj  = makeTensor("4 196 768", 1);
    Tensor* attn_Residual = makeTensorbyShape(dInput, dInput->device_type);//이거는 Attention블럭 들어가기 전에 있어야한다. residual 전달해야됌
    Tensor* dQKV = makeTensor("4 196 2304", 1);
    Tensor* attn_mlp = makeTensor("4 196 3072", 1);


    //start
    dInput = copyTensor(dInput, input);

    //////////////patch embedding///////////////
    O = matmul_bias(O, dInput, EXTRA_weights_d[0], EXTRA_weights_d[1], 0);

    //////////Attention Block////////////


    // O = copyTensor(O, input);
    for(int i=0; i < ATTN_layer_num; i++){
        dMHA_block = copyMHABlock(dMHA_block, MHA_BLOCK[i]);//이거 그냥 다 복사할 것
        //residual store
        copyTensor(attn_Residual, O);
        //normalize 1
        O = normalize(O,O);
        elementWise_Tensor(O, O, '*', dMHA_block[0]);
        elementWise_Tensor(O, O, '+', dMHA_block[1]);

        //dQKV
        dQKV = matmul_bias(dQKV, O, dMHA_block[2], dMHA_block[3], 0);//get QKV
        //flashAttnetion
        O = flashAttention_MHA(O, dQKV);
        //projection
        O_proj = matmul_bias(O_proj, O, dMHA_block[4], dMHA_block[5], 0);
        //residual 1
        O_proj = elementWise_Tensor(O_proj, O_proj, '+', attn_Residual);
        copyTensor(attn_Residual, O_proj);

        //normalize2
        normalize(O_proj, O_proj);
        elementWise_Tensor(O_proj, O_proj, '*', dMHA_block[6]);
        elementWise_Tensor(O_proj, O_proj, '+', dMHA_block[7]);

        //MLP layer
        attn_mlp = matmul_bias(attn_mlp, O_proj, dMHA_block[8], dMHA_block[9], 0);
        attn_mlp = gelu_Tensor(attn_mlp);
        O = matmul_bias(O, attn_mlp,dMHA_block[10], dMHA_block[11], 0);

        //residual 2
        O= elementWise_Tensor(O, O, '+', attn_Residual);
    }
    freeTensor(printTensor(makeSubTensor(copyTensor(makeTensorbyShape(O, 0), O), "0 0 0","8 8")));
    //////////////////////////////////////////////////

    //===========free=================

    freeTensor(O);
    freeMHABlock(dMHA_block);
    freeTensor(O_proj);
    freeTensor(attn_Residual);
    freeTensor(dQKV);
    freeTensor(attn_mlp);

    for(int i=0; i < 12; i++){
        freeMHABlock(MHA_BLOCK[i]);
    }

    freeExtraWeights(EXTRA_weights);
    freeExtraWeights(EXTRA_weights_d);
    freeTensor(input);
    freeTensor(dInput);
}



    
    // ///////ATTNTN////////
    // //residual store
    // copyTensor(attn_Residual, dInput);
    // printf("=input=\n");
    // freeTensor(printTensor(makeSubTensor(copyTensor(makeTensorbyShape(dInput, 0), dInput), "0 0 0","8 8")));
    // //normalize1
    // normalize(dInput, dInput);
    // elementWise_Tensor(dInput, dInput, '*', dMHA_block[0]);//여기의 dMHA_BLOCK은 broadcasting 을 해야한다.
    // printf("=norm=\n");
    // freeTensor(printTensor(makeSubTensor(copyTensor(makeTensorbyShape(dInput, 0), dInput), "0 0 0","8 8")));
    // elementWise_Tensor(dInput, dInput, '+', dMHA_block[1]);

    // freeTensor(printTensor(makeSubTensor(copyTensor(makeTensorbyShape(dInput, 0), dInput), "0 0 0","8 8")));
    // //QKV
    // dQKV = matmul_bias(dQKV, dInput, dMHA_block[2], dMHA_block[3], 0);//get QKV

    // //flashAttention
    // O = flashAttention_MHA(O, dQKV);//Flash Attention
    
    // //projection
    // O_proj = matmul_bias(O_proj, O, dMHA_block[4], dMHA_block[5], 0);//Projection
    // freeTensor(printTensor(makeSubTensor(copyTensor(makeTensorbyShape(O_proj, 0), O_proj), "0 0 0","8 8")));
    
    // //residual 1
    // O_proj = elementWise_Tensor(O_proj, O_proj, '+', attn_Residual);
    // copyTensor(attn_Residual, O_proj);

    // //normalize2
    // normalize(O_proj, O_proj);
    // elementWise_Tensor(O_proj, O_proj, '*', dMHA_block[6]);
    // elementWise_Tensor(O_proj, O_proj, '+', dMHA_block[7]);

    // //MLP layer
    // attn_mlp = matmul_bias(attn_mlp, O_proj, dMHA_block[8], dMHA_block[9], 0);
    // attn_mlp = gelu_Tensor(attn_mlp);
    // O = matmul_bias(O, attn_mlp,dMHA_block[10], dMHA_block[11], 0);

    // //residual 2
    // O_proj = elementWise_Tensor(O, O, '+', attn_Residual);
    // freeTensor(printTensor(makeSubTensor(copyTensor(makeTensorbyShape(O, 0), O), "2 188 760","8 8")));
    // // freeTensor(printTensor(makeSubTensor(copyTensor(makeTensorbyShape(O, 0), O), "2 188 0","8 16")));
    // // infoTensor(dQKV);

