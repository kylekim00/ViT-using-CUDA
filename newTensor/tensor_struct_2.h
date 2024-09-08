#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <cuda_runtime.h>

#define tile_SIZE 8

typedef struct Tensor{
    float *T;                               //이건 배열이 아니라 시작주소이다.
    int *dim;
    int *stride;
    int *d_dim_stride;
    int num_dim;
    int sizeTensor;
    char device_type;
    char isSub;
}Tensor;


// Function Prototypes
Tensor *makeTensor(int *dim, int num_dim, int device_type);
Tensor *makeTensorbyShape(Tensor* src, int device_type);
// Tensor *makeTensorbyCopy(Tensor* src,int *transpose, int device_type);
Tensor *makeSubTensor(Tensor* src, int* start_point, int* dim, int num_dim);

void freeTensor(Tensor *ten);

//reshape
Tensor* reshapeTensor(Tensor* dst, Tensor* src, int* reshape);
Tensor* reshapeTensorinline(Tensor* ten, int* reshape);
Tensor* reshapeTranspose2D(Tensor* ten);


Tensor* copyTensor(Tensor* dst, Tensor* src);

void printTensor(Tensor *ten);
void infoTensor(Tensor *ten);


Tensor* matmul(Tensor* dC, Tensor *dA, Tensor* dB);
Tensor* matmul_matwise(Tensor* dC, Tensor *dA, Tensor* dB);
#endif // TENSOR_H
//주의할 점. 
//어차피 할당된 곳을 계속 쓰게 되어있다. 굳이 free할 일이 거의 없으므로 웬만하면 inline 또는 dst, src 꼴로 만들어 주는 것이 제일 좋다.
