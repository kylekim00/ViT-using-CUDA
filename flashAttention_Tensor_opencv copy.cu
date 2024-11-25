#include<stdio.h>
#include<stdlib.h>
#include<float.h>
#include <curand.h>
#include <curand_kernel.h>
#include "./Easy_Tensor/easy_tensor.h"
#include "MHA.h"
#include<string.h>
#include<time.h>
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>
// #define FOLDER_PATH "./data" // 모니터링할 폴더 경로

#define ATTN_LAYER_NUM 12
int main(){
    char input_dim[] = "4 196 768";
    
    //dummy input
    Tensor* input = makeTensor(input_dim, 0);
    // input = copyTensorfromFILE(input, "floated_img.bin");
    Tensor*dInput = makeTensorbyShape(input, 1);

    Tensor* output = makeTensor("4 1000", 0);
    Tensor* dOutput = makeTensorbyShape(output, 1);

    Tensor* out_argmax = makeTensor("4", 0);
    
    ///////////////pretrained weight initialization////////////////////

    //EXTRAWEIGHTS
    Tensor** extra_weights = makeExtraWeights(0);
    Tensor** extra_weights_d = makeExtraWeights(1);
    
    copyExtraWeightsfromFILE(extra_weights, "extra_weights.bin");
    extra_weights_d = copyExtraWeights(extra_weights_d, extra_weights);

    //MHA_block0 FILE copy

    Tensor** MHA_BLOCK[12];
    Tensor** dMHA_BLOCK[12];
    for(int i=0; i < ATTN_LAYER_NUM; i++){
        MHA_BLOCK[i] = makeMHABlock(0);
        dMHA_BLOCK[i] = makeMHABlock(1);
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

    // Tensor**dMHA_block = makeMHABlock(1);

    for(int i=0; i < ATTN_LAYER_NUM; i++){
        dMHA_BLOCK[i] = copyMHABlock(dMHA_BLOCK[i], MHA_BLOCK[i]);
    }

    for(int i=0; i < ATTN_LAYER_NUM; i++){
        freeMHABlock(MHA_BLOCK[i]);
    }

    ///////////////////////TMP TENSORS////////////////////////////
    

    //Attention tmp tensors.

    Tensor* dInput_embed = makeTensorbyShape(input, 1);
    
    Tensor* O = makeTensor("4 197 768", 1);
    Tensor* O_proj = makeTensor("4 197 768", 1);
    Tensor* attn_Residual = makeTensorbyShape(O, 1);//이거는 Attention블럭 들어가기 전에 있어야한다. residual 전달해야됌
    Tensor* dQKV = makeTensor("4 197 2304", 1);
    Tensor* attn_mlp = makeTensor("4 197 3072", 1);

    Tensor* head = makeTensor("4 768", 1);
    
    ////////////////////////initialization////////////////////////////////////////



    //////////////////////////start of Iteration//////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////
    //
    clock_t st_time = clock();

    for(int iteration = 0; iteration < 10; iteration++){
        printf("%d\n", iteration);
    //
    //////////////////////////////////////////////////////////////////////////////
    input = copyTensorfromFILE(input, "data_queue/floated_img.bin");

    dInput = copyTensor(dInput,input);
    
    //////////////patch embedding///////////////
    dInput_embed = matmul_bias(dInput_embed, dInput, extra_weights_d[0], extra_weights_d[1], 0);
    
    ///////////////cls_token////////////////////
    O = add_CLS_token_init(O, extra_weights_d[2]);//cls token 넣기
    O = add_CLS_token(O, dInput_embed);                  //input넣기
    
    //////////////pos_embed/////////////////////
    O = elementWise_Tensor(O, O, '+', extra_weights_d[3]);
    // O = copyTensor(O, dInput);//임시 197

    //////////Attention Block////////////


    // O = copyTensor(O, input);
    for(int i=0; i < ATTN_LAYER_NUM; i++){
        // dMHA_block = copyMHABlock(dMHA_block, MHA_BLOCK[i]);//이거 그냥 다 복사할 것
        //residual store
        copyTensor(attn_Residual, O);
        //normalize 1
        O = normalize(O,O);
        elementWise_Tensor(O, O, '*', dMHA_BLOCK[i][0]);
        elementWise_Tensor(O, O, '+', dMHA_BLOCK[i][1]);

        //dQKV
        // dQKV = matmul_bias(dQKV, O, dMHA_BLOCK[i][2], dMHA_BLOCK[i][3], 0);//get QKV
        matmul_cublas_batched_bias(dQKV, O, dMHA_BLOCK[i][2], dMHA_BLOCK[i][3]);
        //flashAttnetion
        O = flashAttention_MHA(O, dQKV);
        //projection
        // O_proj = matmul_bias(O_proj, O, dMHA_BLOCK[i][4], dMHA_BLOCK[i][5], 0);
        matmul_cublas_batched_bias(O_proj, O, dMHA_BLOCK[i][4], dMHA_BLOCK[i][5]);
        //residual 1
        O_proj = elementWise_Tensor(O_proj, O_proj, '+', attn_Residual);
        copyTensor(attn_Residual, O_proj);

        //normalize2
        normalize(O_proj, O_proj);
        elementWise_Tensor(O_proj, O_proj, '*', dMHA_BLOCK[i][6]);
        elementWise_Tensor(O_proj, O_proj, '+', dMHA_BLOCK[i][7]);

        //MLP layer
        // attn_mlp = matmul_bias(attn_mlp, O_proj, dMHA_BLOCK[i][8], dMHA_BLOCK[i][9], 0);
        matmul_cublas_batched_bias(attn_mlp, O_proj, dMHA_BLOCK[i][8], dMHA_BLOCK[i][9]);
        attn_mlp = gelu_Tensor(attn_mlp);
        // O = matmul_bias(O, attn_mlp,dMHA_BLOCK[i][10], dMHA_BLOCK[i][11], 0);
        matmul_cublas_batched_bias(O, attn_mlp,dMHA_BLOCK[i][10], dMHA_BLOCK[i][11]);

        //residual 2
        O = elementWise_Tensor(O, O, '+', attn_Residual);
    }
    //////////put head out////////////

    //MLP head
    head = cut_HEAD_out(head, O);

    //normalization
    head = normalize(head, head);
    head = elementWise_Tensor(head, head, '*', extra_weights_d[4]);
    head = elementWise_Tensor(head, head, '+', extra_weights_d[5]);
    //mlp(768, 1000)
    dOutput = matmul_bias(dOutput, head, extra_weights_d[6], extra_weights_d[7], 0);
    matmul_cublas_batched_bias(dOutput, head, extra_weights_d[6], extra_weights_d[7]);
    output = copyTensor(output, dOutput);
    out_argmax = maxmax(out_argmax, output);
    for(int i=0; i < out_argmax->dim[0]; i++){
        printf("%s\n", IMAGENET_LABELS[(int)out_argmax->T[i]]);
    }
    ////////////////////////////end of Iteration//////////////////////////////////
    //
    // printTensor(out_argmax);
    }
    
    clock_t end_time = clock();
    double time_taken = (double)(end_time - st_time) / CLOCKS_PER_SEC;
    printf("실행 시간: %f 초\n", time_taken);
    //////////////////////////////////////////////////////////////////////////////
    freeTensor(printTensor(makeSubTensor(output, "0 895","4 8")));
    //////////////////////////////////////////////////

    //===========free=================

    freeTensor(O);
    freeTensor(O_proj);
    freeTensor(attn_Residual);
    freeTensor(dQKV);
    freeTensor(attn_mlp);

    for(int i=0; i < ATTN_LAYER_NUM; i++){
        freeMHABlock(dMHA_BLOCK[i]);
    }
    freeExtraWeights(extra_weights);
    freeExtraWeights(extra_weights_d);
    
    freeTensor(output);
    freeTensor(dOutput);
    freeTensor(out_argmax);
    freeTensor(input);
    freeTensor(dInput);
}

