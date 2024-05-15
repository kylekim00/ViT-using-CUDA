//nvcc  tiled_matmul_test.cu matrix_struct.cu -I.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "matrix_struct.h"

double generateStandardNormal() {
    double u1, u2, w, mult;
    static double x1, x2;
    static int call = 0;

    if (call == 1) {
        call = !call;
        return x2;
    }

    do {
        u1 = 2.0 * ((double) rand() / RAND_MAX) - 1.0;
        u2 = 2.0 * ((double) rand() / RAND_MAX) - 1.0;
        w = u1 * u1 + u2 * u2;
    } while (w >= 1.0 || w == 0);

    mult = sqrt((-2.0 * log(w)) / w);
    x1 = u1 * mult;
    x2 = u2 * mult;
    call = !call;

    return x1;
}

void dummyMatrix(Matrix *mat){
    for(int i=0; i < mat->row * mat-> col; i++){
        mat->M[i] = i-mat->row * mat-> col;
    }
}

#define NUM_HIDDEN_LAYER 3

int main(){
    // srand(10); // Seed the random number generator

    // for (int i = 0; i < 10; i++) {
    //     printf("%f\n", generateStandardNormal());
    // }
    ////////////////////////////////////////////////////////
    int batch_size = 4;
    int input_Layer = 512;
    int output_Layer = 8;
    int hidden_Layer_size[NUM_HIDDEN_LAYER] = {64, 32, 40};
    //////////////////////// DATA LABEL ALLOCATION /////////////////////////////////

    Matrix *input;//우리의 데이터. // forward and backward
    Matrix *label;
    

    input = makeMatrix(batch_size, input_Layer, 0);//input 공간할당
    dummyMatrix(input);
    label = makeMatrix(batch_size, 1, 0);
    

    /////////////////////// WEIGHT BIAS ALLOCATION /////////////////////////////

    Matrix *W[NUM_HIDDEN_LAYER+1];   //W Matrix 배열
    Matrix *B[NUM_HIDDEN_LAYER+1];
    
    W[0] = makeMatrix(input_Layer, hidden_Layer_size[0], 0);//W[0] alloc
    B[0] = makeMatrix(1, hidden_Layer_size[0], 0);
    for(int i=1; i < NUM_HIDDEN_LAYER; i++){
        W[i] = makeMatrix(hidden_Layer_size[i-1], hidden_Layer_size[i], 0);//W[hidden] alloc
        B[i] = makeMatrix(1, hidden_Layer_size[i], 0);
        dummyMatrix(W[i]);dummyMatrix(B[i]);//가중치 추가
    }
    W[NUM_HIDDEN_LAYER] = makeMatrix(hidden_Layer_size[NUM_HIDDEN_LAYER-1],output_Layer, 0);//W[n-1] alloc
    B[NUM_HIDDEN_LAYER] = makeMatrix(1, output_Layer, 0);

    ///////////////////// WEIGHT BIAS INITIALIZATION /////////////////////////////

    //Weights dimension check
    printf("=====W dimension=====\n");
    for(int i=0; i < NUM_HIDDEN_LAYER+1;i++){
        printf("W[%d] : ",i);
        infoMatrix(W[i]);
        printf("B[%d] : ",i);
        infoMatrix(B[i]);
    }
    printf("=====================\n");
    
    ///////////////////////////////DATA TRANSFER FROM FILE/////////////////////////////////
    dummyMatrix(input);
    //////////////////////////////device memory alloc/////////////////////////////////////
    Matrix *dInput = moveMatrix(input, 1);

    Matrix *dW[NUM_HIDDEN_LAYER + 1];//W
    Matrix *dA[NUM_HIDDEN_LAYER + 1];//activation function layer
    
    Matrix *dO;//last softmax layer
    Matrix *dB[NUM_HIDDEN_LAYER + 1];//bias


    //W & B memory copy[0~3]
    dW[0] = copyMatrix(W[0], 1);// 이거 move로 바꿔도 되는 거 아닌가?????
    dB[0] = copyMatrix(B[0], 1);
    for(int i=1; i <= NUM_HIDDEN_LAYER; i++){
        dW[i] = copyMatrix(W[i], 1);
        dB[i] = copyMatrix(B[i], 1);
    }


    for(int i=0; i< sizeof(dW)/sizeof(Matrix*); i++){
        infoMatrix(dW[i]);
    }
    ////////////////////////////////=FORWARD PASS=/////////////////////////////////
    
    //나중에 메모리를 해제하는 것은 dA만으로 충분하다.
    dA[0] = ReLU(matmul_Bias(dInput, dW[0], dB[0]));
    for(int i=1; i < NUM_HIDDEN_LAYER; i++){
        dA[i] = ReLU(matmul_Bias(dA[i-1], dW[i], dB[i]));//여기 계속 make하네 꼭 iteration마다 해제 해줘야함.
    }
    dA[NUM_HIDDEN_LAYER] = matmul_Bias(dA[NUM_HIDDEN_LAYER-1], dW[NUM_HIDDEN_LAYER], dB[NUM_HIDDEN_LAYER]);//마지막 레이어는 activation 을 통과하면 안됨.
    
    // softMax_Rowwise(dA[NUM_HIDDEN_LAYER], dO);//이거 디바이스 땜에 오류남. 근데 어차피 디바이스에 라벨도 다 올리는데 그냥 디바이스에서 하는게 맞지 않을까.
    

    //dim check
    printf("=======dA dimension=========\n");
    for(int i=0; i < sizeof(dA)/sizeof(Matrix*);i++){
        infoMatrix(dA[i]);
    }
    dO = copyMatrix(dA[NUM_HIDDEN_LAYER], 0);
    printf("=======dO dimemtion=========\n");
    printMatrix(dO);
    softMax_Rowwise_inline(dO,dO);
    infoMatrix(dO);
    printMatrix(dO);

    ////////////////////////////////=LOSS CALCULATION=/////////////////////////////////
    


    ////////////////////////////////=BACKWARD PASS=/////////////////////////////////

    return 0;
}