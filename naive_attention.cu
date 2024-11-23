#include<stdio.h>
#include<stdlib.h>
#include<float.h>
#include <curand.h>
#include <curand_kernel.h>
#include<cuda_runtime.h>
#include "./Easy_Tensor/easy_tensor.h"
#include "MHA.h"
#include<string.h>
#include<time.h>

#define ATTN_LAYER_NUM 12

Tensor* dummyTensor(Tensor *ten){
    for(int i=0; i < ten->dim[0] * ten->stride[0]; i++){
        ten->T[i] = i;
    }
    return ten;
}



Tensor* naiveAttention(Tensor* O, Tensor* dQKV){
    Tensor* dQKV_tmp = makeTensor("4 197 3 12 64", 1);
    cudaMemcpy(dQKV_tmp->T, dQKV->T, dQKV->sizeTensor * sizeof(float), cudaMemcpyDeviceToDevice);

    Tensor* dQKV_T = makeTensor("3 4 12 197 64", 1);//QKV, batch, head, patch, vector

    int reshape[] = {2, 0, 3, 1, 4};
    copyReshapeTensor(dQKV_T, dQKV_tmp, reshape);

    freeTensor(dQKV_tmp);

    Tensor* dQ = makeTensor("4 12 197 64", 1);
    Tensor* dK = makeTensor("4 12 197 64", 1);
    Tensor* dV = makeTensor("4 12 197 64", 1);
//====
    int i = 0;
    cudaMemcpy(dQ->T, dQKV_T->T, dQ->sizeTensor * sizeof(float), cudaMemcpyDeviceToDevice);
    i += dQKV_T->stride[0];
    cudaMemcpy(dK->T, (dQKV_T->T + i), dQ->sizeTensor * sizeof(float), cudaMemcpyDeviceToDevice);
    i += dQKV_T->stride[0];
    cudaMemcpy(dV->T, dQKV_T->T + i, dQ->sizeTensor * sizeof(float), cudaMemcpyDeviceToDevice);
    freeTensor(dQKV_T);
    dQ = scalar_Tensor(dQ, '*',0.125);
    Tensor* dKt = makeTensor("4 12 64 197", 1);
    copyTransposeTensor(dKt, dK);
    freeTensor(dK);

    Tensor* QKt = makeTensor("4, 12, 197, 197", 1);

    matmul(QKt, dQ, dKt);
    freeTensor(dQ);
    // freeTensor(dKt);

    softMax_broad(QKt, QKt);

    Tensor* dO_mini = makeTensor("4 12 197, 64", 1);
    matmul(dO_mini, QKt, dV);
    // freeTensor(QKt);
    freeTensor(dV);

    Tensor* dO = makeTensor("4 197 12 64", 1);
    int reshape2[] = {0, 2, 1, 3};
    copyReshapeTensor(dO, dO_mini, reshape2);
    freeTensor(dO_mini);

    cudaMemcpy(O->T, dO->T, dO->sizeTensor * sizeof(float), cudaMemcpyDeviceToDevice);
    freeTensor(dO);

    return dKt;
}





int main(){
    Tensor* dQKV = dummyTensor(makeTensor("4 197 2304", 0));
    Tensor* dQKV_d = copyTensor(makeTensorbyShape(dQKV, 1), dQKV);
    // printTensor(makeSubTensor(copyTensor(makeTensorbyShape(tmp, 0),tmp), "0 1 0 0 0", "64"));

    Tensor* O = makeTensor("4 197 768", 1);


    // Tensor* tmp = naiveAttention(O, dQKV_d);

    printTensor(makeSubTensor(copyTensor(makeTensorbyShape(O, 0), O), "0 1 0", "64"));
    // printTensor(makeSubTensor(copyTensor(makeTensorbyShape(tmp, 0), tmp), "1 1 56 186", "8 8"));
    flashAttention_MHA(O, dQKV_d);
    printTensor(makeSubTensor(copyTensor(makeTensorbyShape(O, 0), O), "0 1 0", "64"));
    
    Tensor* A = dummyTensor(makeTensor("4 3 5 5", 0));
    Tensor* B = dummyTensor(makeTensor("4 3 5 5", 0));
    Tensor* dA = makeTensorbyShape(A, 1);
    Tensor* dB = makeTensorbyShape(B, 1);
    Tensor* tmp = makeTensor("4 3 5 5", 0);
    Tensor* d_tmp = makeTensorbyShape(tmp, 1);
    
    printTensor(copyTensor(tmp, matmul(d_tmp, dA, dB)));

    
    // printTensor(tmp);



    // char input_dim[] = "4 196 768";
    // //dummy input
    // Tensor* input = makeTensor(input_dim, 0);
    // input = copyTensorfromFILE(input, "dummy_input_4_196_768.bin");
    // Tensor*dInput = makeTensorbyShape(input, 1);

    // Tensor* output = makeTensor("4 1000", 0);
    // Tensor* dOutput = makeTensorbyShape(output, 1);

    // Tensor* out_argmax = makeTensor("4", 0);
    
    // ///////////////pretrained weight initialization////////////////////

    // //EXTRAWEIGHTS
    // Tensor** extra_weights = makeExtraWeights(0);
    // Tensor** extra_weights_d = makeExtraWeights(1);
    
    // copyExtraWeightsfromFILE(extra_weights, "extra_weights.bin");
    // extra_weights_d = copyExtraWeights(extra_weights_d, extra_weights);

    // //MHA_block0 FILE copy

    // Tensor** MHA_BLOCK[12];
    // Tensor** dMHA_BLOCK[12];
    // for(int i=0; i < ATTN_LAYER_NUM; i++){
    //     MHA_BLOCK[i] = makeMHABlock(0);
    //     dMHA_BLOCK[i] = makeMHABlock(1);
    // }



    // copyMHABlockfromFILE(MHA_BLOCK[0], "0_newblock.bin");
    // copyMHABlockfromFILE(MHA_BLOCK[1], "1_newblock.bin");
    // copyMHABlockfromFILE(MHA_BLOCK[2], "2_newblock.bin");
    // copyMHABlockfromFILE(MHA_BLOCK[3], "3_newblock.bin");
    // copyMHABlockfromFILE(MHA_BLOCK[4], "4_newblock.bin");
    // copyMHABlockfromFILE(MHA_BLOCK[5], "5_newblock.bin");
    // copyMHABlockfromFILE(MHA_BLOCK[6], "6_newblock.bin");
    // copyMHABlockfromFILE(MHA_BLOCK[7], "7_newblock.bin");
    // copyMHABlockfromFILE(MHA_BLOCK[8], "8_newblock.bin");
    // copyMHABlockfromFILE(MHA_BLOCK[9], "9_newblock.bin");
    // copyMHABlockfromFILE(MHA_BLOCK[10], "10_newblock.bin");
    // copyMHABlockfromFILE(MHA_BLOCK[11], "11_newblock.bin");

    // // Tensor**dMHA_block = makeMHABlock(1);

    // for(int i=0; i < ATTN_LAYER_NUM; i++){
    //     dMHA_BLOCK[i] = copyMHABlock(dMHA_BLOCK[i], MHA_BLOCK[i]);
    // }

    // for(int i=0; i < ATTN_LAYER_NUM; i++){
    //     freeMHABlock(MHA_BLOCK[i]);
    // }

    // ///////////////////////TMP TENSORS////////////////////////////
    

    // //Attention tmp tensors.

    // Tensor* dInput_embed = makeTensorbyShape(input, 1);
    
    // Tensor* O = makeTensor("4 197 768", 1);
    // Tensor* O_proj = makeTensor("4 197 768", 1);
    // Tensor* attn_Residual = makeTensorbyShape(O, 1);//이거는 Attention블럭 들어가기 전에 있어야한다. residual 전달해야됌
    // Tensor* dQKV = makeTensor("4 197 2304", 1);
    // Tensor* attn_mlp = makeTensor("4 197 3072", 1);

    // Tensor* head = makeTensor("4 768", 1);
    
    // ////////////////////////initialization////////////////////////////////////////



    // //////////////////////////start of Iteration//////////////////////////////////
    // //////////////////////////////////////////////////////////////////////////////
    // //
    // clock_t st_time = clock();

    // for(int iteration = 0; iteration < 100; iteration++){
    //     printf("%d\n", iteration);
    // //
    // //////////////////////////////////////////////////////////////////////////////

    // dInput = copyTensor(dInput,input);

    // //////////////patch embedding///////////////
    // dInput_embed = matmul_bias(dInput_embed, dInput, extra_weights_d[0], extra_weights_d[1], 0);
    
    // ///////////////cls_token////////////////////
    // O = add_CLS_token_init(O, extra_weights_d[2]);//cls token 넣기
    // O = add_CLS_token(O, dInput_embed);                  //input넣기
    
    // //////////////pos_embed/////////////////////
    // O = elementWise_Tensor(O, O, '+', extra_weights_d[3]);
    // // O = copyTensor(O, dInput);//임시 197

    // //////////Attention Block////////////


    // // O = copyTensor(O, input);
    // for(int i=0; i < ATTN_LAYER_NUM; i++){
    //     // dMHA_block = copyMHABlock(dMHA_block, MHA_BLOCK[i]);//이거 그냥 다 복사할 것
    //     //residual store
    //     copyTensor(attn_Residual, O);
    //     //normalize 1
    //     O = normalize(O,O);
    //     elementWise_Tensor(O, O, '*', dMHA_BLOCK[i][0]);
    //     elementWise_Tensor(O, O, '+', dMHA_BLOCK[i][1]);

    //     //dQKV
    //     dQKV = matmul_bias(dQKV, O, dMHA_BLOCK[i][2], dMHA_BLOCK[i][3], 0);//get QKV
    //     //flashAttnetion
    //     // O = flashAttention_MHA(O, dQKV);
    //     O = naiveAttention(O, dQKV);
    //     //projection
    //     O_proj = matmul_bias(O_proj, O, dMHA_BLOCK[i][4], dMHA_BLOCK[i][5], 0);
    //     //residual 1
    //     O_proj = elementWise_Tensor(O_proj, O_proj, '+', attn_Residual);
    //     copyTensor(attn_Residual, O_proj);

    //     //normalize2
    //     normalize(O_proj, O_proj);
    //     elementWise_Tensor(O_proj, O_proj, '*', dMHA_BLOCK[i][6]);
    //     elementWise_Tensor(O_proj, O_proj, '+', dMHA_BLOCK[i][7]);

    //     //MLP layer
    //     attn_mlp = matmul_bias(attn_mlp, O_proj, dMHA_BLOCK[i][8], dMHA_BLOCK[i][9], 0);
    //     attn_mlp = gelu_Tensor(attn_mlp);
    //     O = matmul_bias(O, attn_mlp,dMHA_BLOCK[i][10], dMHA_BLOCK[i][11], 0);

    //     //residual 2
    //     O = elementWise_Tensor(O, O, '+', attn_Residual);
    // }
    // //////////put head out////////////

    // //MLP head
    // head = cut_HEAD_out(head, O);

    // //normalization
    // head = normalize(head, head);
    // head = elementWise_Tensor(head, head, '*', extra_weights_d[4]);
    // head = elementWise_Tensor(head, head, '+', extra_weights_d[5]);
    // //mlp(768, 1000)
    // dOutput = matmul_bias(dOutput, head, extra_weights_d[6], extra_weights_d[7], 0);

    // output = copyTensor(output, dOutput);
    // out_argmax = maxmax(out_argmax, output);
    // ////////////////////////////end of Iteration//////////////////////////////////
    // //
    // // printTensor(out_argmax);
    // }
    
    // clock_t end_time = clock();
    // double time_taken = (double)(end_time - st_time) / CLOCKS_PER_SEC;
    // printf("실행 시간: %f 초\n", time_taken);
    // //////////////////////////////////////////////////////////////////////////////
    // freeTensor(printTensor(makeSubTensor(output, "0 18","4 8")));
    // //////////////////////////////////////////////////

    // //===========free=================

    // freeTensor(O);
    // freeTensor(O_proj);
    // freeTensor(attn_Residual);
    // freeTensor(dQKV);
    // freeTensor(attn_mlp);

    // for(int i=0; i < ATTN_LAYER_NUM; i++){
    //     freeMHABlock(dMHA_BLOCK[i]);
    // }
    // freeExtraWeights(extra_weights);
    // freeExtraWeights(extra_weights_d);
    
    // freeTensor(output);
    // freeTensor(dOutput);
    // freeTensor(out_argmax);
    // freeTensor(input);
    // freeTensor(dInput);
}

