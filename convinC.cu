#include<stdio.h>
#include<stdlib.h>
#include<float.h>
#include<matrix_struct.h>

// typedef struct Matrix{
//     float *M;
//     int row;
//     int col;
// }Matrix;
#define tile_SIZE 8


Matrix* convtoMatMul(Matrix *M, int input_size, int kernal_size, int stride, int out_channel){
    int output_size = (input_size - kernal_size)/stride + 1;

    Matrix *convM = makeMatrix(output_size * output_size, kernal_size * kernal_size, 0);
    for(int i=0; i < output_size; i++){
        for(int j=0; j < output_size; j++){
            for(int k=0; k < kernal_size; k++){
                for(int l = 0; l < kernal_size; l++){
                    if((k + i * stride < input_size) && (l + j * stride < input_size)){
                        convM->M[(i*output_size+j)*convM->col+k*kernal_size+l] = M->M[(k+i*stride)*M->col+j*stride+l];
                    }
                    else{
                        convM->M[(i*output_size+j)*convM->col+k*kernal_size+l] = 0;
                    }
                }
            }
        }
    }
    return convM;
}
Matrix* dummyMatrix(Matrix *mat){
    for(int i=0; i < mat->row * mat-> col; i++){
        float dm = i;
        mat->M[i] = dm;
    }
    return mat;
}
int main(){
    int input_size = 16;
    int kernal_size = 3;
    int stride = 2;
    int out_channel = 3;
    Matrix *A = makeMatrix(16, 16, 0);
    dummyMatrix(A);
    printMatrix(convtoMatMul(A, input_size,kernal_size, stride, out_channel));
    printf("\nmin:%lf", -(__DBL_MAX__));
    
}