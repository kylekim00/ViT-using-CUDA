#include<stdio.h>
#include<stdlib.h>
#include "tensor_struct.h"

Tensor* dummyTensor(Tensor *ten){
    for(int i=0; i < ten->dim[0] * ten->stride[0]; i++){
        float dm = 2;
        ten->T[i] = dm;
    }
    return ten;
}
void printfirstMatinTensor(Tensor *ten){
    if(ten->device_type){
        printf("printTensor : GPU mem can not be printed\n");
        return;
    }
    //==============if ten->num_dim < 3========================
    //=================else====================================
    printf("=\n");
    int numofMat = ten->dim[0] * ten->stride[0] / ten->stride[ten->num_dim-3];
    // for(int n=0; n < ten->num_dim - 2; n++){
    //     numofMat *= dim[n];
    // }
    for(int mat_inx = 0; mat_inx < 1; mat_inx++){
        printf("[ ");
        int tmp = mat_inx;
        for(int n=0; n < ten->num_dim - 3; n++){
            printf("%d, ", tmp / (ten->stride[n]/ten->stride[ten->num_dim-3]));
            tmp %= (ten->stride[n]/ten->stride[ten->num_dim-3]);
        }
        printf("%d, ", tmp);
        printf("-, -]\n");
        for(int i = mat_inx * ten->stride[ten->num_dim-3]; i < mat_inx * ten->stride[ten->num_dim-3] + ten->stride[ten->num_dim-3]; i+=ten->stride[ten->num_dim-2]){
            for(int j= i; j < i + ten->stride[ten->num_dim - 2];j++){
                printf("%.02f\t", ten->T[j]);
            }
            printf("\n");
        }
    }
}   
int main(){
    int dim[] = {4, 1, 196, 768};
    int dim2[] = {1, 768, 768};
    Tensor *A = makeTensor(dim, sizeof(dim)/sizeof(int), 0);
    Tensor *At = makeTensor(dim2, sizeof(dim2)/sizeof(int), 0);
    
}