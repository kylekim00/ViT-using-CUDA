#ifndef MHA_H
#define MHA_H

#include <stdio.h>
#include <cuda_runtime.h>
#include "./Easy_Tensor/easy_tensor.h"

Tensor** makeExtraWeights(int device_type);
void freeExtraWeights(Tensor** block);
Tensor** copyExtraWeights(Tensor** dst, Tensor** src);
Tensor** copyExtraWeightsfromFILE(Tensor** block, const char* file_name);

Tensor** makeMHABlock(int device_type);
void freeMHABlock(Tensor** block);
Tensor** copyMHABlockfromFILE(Tensor** block, const char* file_name);


Tensor* flashAttention_MHA(Tensor* O, Tensor* dQKV);

Tensor* copyTensorfromFILE(Tensor* dst, const char* file_name);

Tensor** copyMHABlock(Tensor** dst, Tensor** src);

Tensor* add_CLS_token(Tensor* input_197_d, Tensor* input_d);

Tensor* add_CLS_token_init(Tensor* input_197_d, Tensor* cls_token);

Tensor* cut_HEAD_out(Tensor*head, Tensor* O);

Tensor* maxmax(Tensor* arg_max_CPU, Tensor*src_CPU);
#endif // TENSOR_H
