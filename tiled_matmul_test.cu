//nvcc  tiled_matmul_test.cu matrix_struct.cu -I.
#include "matrix_struct.h"
#include<unistd.h>
#include<cuda_runtime.h>
#include <stdio.h>
Matrix* dummyMatrix(Matrix *mat){
    for(int i=0; i < mat->row * mat-> col; i++){
        float dm = i*i/10;
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
    printMatrix(A);
    Matrix *dA = moveMatrix(A, 1);
    dA = softMax_Rowwise_inline(dA, dA);
    A = moveMatrix(dA, 0);
    printMatrix(A);   

    // dummyMatrix(B);
    // printMatrix(A);
    // printf("==\n");
    // Matrix *dA = copyMatrix(A, 1);
    // printMatrix(moveMatrix(softMax_Rowwise_inline(dA, dA), 0));
    // Matrix *B = makeMatrix(2, 3, 0);
    // dummyMatrix(B);

    // Matrix *dA = moveMatrix(A, 1);
    // printf("%f", A->M[3]);
    // Matrix *dB = copyMatrix(B, 1);
    // Matrix *dC = matmul(dA,dB);
    // Matrix *C = moveMatrix(dC,0);
}

//쿠다를 해제한다고 해서 할당된 메모리가 그냥 의미없이 사라지는 것이 아니다. 마치 휴지통에 지운다고 해서
//그 메모리를 못쓰는게 아닌것 처럼. 그냥 덮어서 쓸 뿐이다. 그래서 free를 시켜도 할당된 메모리 사용랑이 변하지 않는 것이다. 