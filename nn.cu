#include<stdio.h>
#include<stdlib.h>
#include "easy_tensor.h"

Tensor* copyTensorfromFILE(Tensor* dst, char* file_name);

int main(){
    Tensor* input_batch = makeTensor("4 1 784", 0);
    infoTensor(input_batch);
    Tensor* W[4];
    W[0] = makeTensor("784 50", 0);
    W[1] = makeTensor("50 30", 0);
    W[2] = makeTensor("30 40", 0);
    W[3] = makeTensor("40 10", 0);
    for(int i=0; i < sizeof(W)/sizeof(Tensor*); i++){
        infoTensor(W[i]);
    }

}

Tensor* copyTensorfromFILE(Tensor* dst, char* file_name){
    char f_name[50] = "./pre_weights/";
    for(int i=0; file_name[i]; i++){
        f_name[i+14] = file_name[i];
        f_name[i+15] = 0;
    }
    
    FILE *file = fopen(f_name, "rb");
    if (!file) {
        printf("Error opening file\n");
        return NULL;
    }

    size_t num_elements = fread(dst->T, sizeof(float), dst->dim[0]*dst->stride[0], file);
    if (num_elements != dst->dim[0]*dst->stride[0]) {
        printf("Error reading file\n");
        return NULL;
    }

    fclose(file);

    return dst;
}