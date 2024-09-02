#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include "tensor_struct.h"

#define NUM_HIDDEN_LAYER 3
int main(){
    int batch_size = 4;
    int input_Layer = 400;
    int output_Layer = 10;
    int hidden_Layer_size[NUM_HIDDEN_LAYER] = {50, 30, 40};

    Tensor *input;
    Tensor *label;
    
}