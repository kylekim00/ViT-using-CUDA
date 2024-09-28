#ifndef MHA_H
#define MHA_H

#include <stdio.h>
#include <cuda_runtime.h>
#include "easy_tensor.h"


Tensor** makeMHABlock(int device_type);
void freeMHABlock(Tensor** block);
Tensor** copyMHABlockfromFILE(Tensor** block, const char* file_name);


Tensor* flashAttention_MHA(Tensor* O, Tensor* dQKV);

Tensor* copyTensorfromFILE(Tensor* dst, const char* file_name);

Tensor** copyMHABlock(Tensor** dst, Tensor** src);

#endif // TENSOR_H
