//nvcc  tiled_matmul_test.cu matrix_struct.cu -I.
#include "matrix_struct.h"
#include<unistd.h>
#include<cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

Matrix* dummyMatrix(Matrix *mat){
    for(int i=0; i < mat->row * mat-> col; i++){
        float dm = 0.1 *i*i;
        mat->M[i] = dm;
    }
    return mat;
}
void zeroMatrix(Matrix *mat){
    for(int i=0; i < mat->row * mat-> col; i++){
        mat->M[i] = 0;
    }
}
int main(){
    // //Memory Test
    // Matrix *A = makeMatrix(10, 10, 0);
    // for(int i=0; i < 10; i++){
    //     A = moveMatrix(A, 1);//사용하는 GPU 메모리의 크기가 커지지 않는다.
    //     // sleep(5);
    //     A = moveMatrix(A, 0);
    // }


    Matrix *A = makeMatrix(5, 7, 0);
    dummyMatrix(A);
    Matrix *B = copyMatrix(makeMatrix(5, 7, 1), A);
    B = softMax_Rowwise_inline(B, B);
    Matrix *C = makeMatrix(5, 7, 1);
    copyMatrix(C, B);
    copyMatrix(A, C);
    printMatrix(A);
    // Matrix *C = copyMatrix(makeMatrix(5, 7, 2), B);
    // A = copyMatrix(A, C);
    // printMatrix(A);

    // printMatrix(copyMatrix(makeMatrix(B->col, B->row, 0) , transposeMatrix(makeMatrix(B->col, B->row, 1), B)));  //이렇게 할 경우 conflict 발생 가능 이거 생각해야함. 
    // cudaCheckError(cudaGetLastError());
    // cudaCheckError(cudaDeviceSynchronize());
}

//쿠다를 해제한다고 해서 할당된 메모리가 그냥 의미없이 사라지는 것이 아니다. 마치 휴지통에 지운다고 해서
//그 메모리를 못쓰는게 아닌것 처럼. 그냥 덮어서 쓸 뿐이다. 그래서 free를 시켜도 할당된 메모리 사용랑이 변하지 않는 것이다. 
//걍 더 쓸 때 더 늘어날 뿐이다.
