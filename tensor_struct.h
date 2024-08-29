#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <cuda_runtime.h>

#define tile_SIZE 8

typedef struct Tensor{
    float *T;
    int *dim;
    int *stride;
    int num_dim;
    char device_type;
}Tensor;


// Function Prototypes
Tensor *makeTensor(int *dim, int num_dim, int device_type);
void freeTensor(Tensor *ten);

void printTensor(Tensor *ten);

Tensor* copyTensor(Tensor* dst, Tensor* src);
Tensor* matmul(Tensor* dC, Tensor *dA, Tensor* dB);
#endif // TENSOR_H
