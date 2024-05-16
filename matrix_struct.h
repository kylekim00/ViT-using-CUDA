#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <cuda_runtime.h>

#define tile_SIZE 4

typedef struct Matrix {
    char device_type;  // 0 for CPU, non-zero for GPU device ID
    int row;
    int col;
    float *M;
} Matrix;


// Function Prototypes
Matrix* makeMatrix(int row, int col, int device_type);  //make matrix and allocate memory
void freeMatrix(Matrix *mat);                           //free memory
void printMatrix(Matrix *mat);                          //print CPU matrix
Matrix* copyMatToDevice(Matrix *mat, int device_type);  //deep copy matrix to other device
Matrix* copyMatToHost(Matrix *dMat);
Matrix* copyMatrix(Matrix *mat, int device_type);  //deep copy matrix to other device
// Matrix* matmul(Matrix *dA, Matrix *dB);
Matrix *matmul(Matrix*dA, Matrix *dB);
Matrix *matmul_inline(Matrix*dC, Matrix*dA, Matrix *dB);
Matrix *matmul_Bias(Matrix*dA, Matrix *dB, Matrix *dBias);
Matrix *matmul_Bias_inline(Matrix*res, Matrix*dA, Matrix *dB, Matrix *dBias);
Matrix* moveMatrix(Matrix *mat, int device_type);
Matrix* ReLU_inline(Matrix *mat);
void infoMatrix(Matrix *mat);
Matrix *softMax_Rowwise_inline(Matrix *dRes, Matrix *dMat);
Matrix *matAdd(Matrix *dMat, Matrix *dA, Matrix *dB);
Matrix *matSub(Matrix *dMat, Matrix *dA, Matrix *dB);
#endif // MATRIX_H
