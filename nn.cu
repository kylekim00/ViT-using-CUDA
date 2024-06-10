//nvcc  tiled_matmul_test.cu matrix_struct.cu -I.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "matrix_struct.h"

#define NUM_HIDDEN_LAYER 3



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
        mat->M[i] = generateStandardNormal()/5;
    }
}


int main(){
    // srand(10); // Seed the random number generator

    // for (int i = 0; i < 10; i++) {
    //     printf("%f\n", generateStandardNormal());
    // }
    ////////////////////////////////////////////////////////
    int batch_size = 4;
    int input_Layer = 400;
    int output_Layer = 10;
    int hidden_Layer_size[NUM_HIDDEN_LAYER] = {50, 30, 40};
//////////////////////// DATA LABEL ALLOCATION /////////////////////////////////

    Matrix *input;//우리의 데이터. // forward and backward
    Matrix *label;

    input = makeMatrix(batch_size, input_Layer, 0);//input 공간할당
    label = makeMatrix(1, batch_size, 0);//label 공간할당.


/////////////////////// WEIGHT BIAS ALLOCATION /////////////////////////////

    Matrix *W[NUM_HIDDEN_LAYER+1];   //W Matrix 배열
    Matrix *B[NUM_HIDDEN_LAYER+1];
    Matrix *O;


    W[0] = makeMatrix(input_Layer, hidden_Layer_size[0], 0);//W[0] alloc
    B[0] = makeMatrix(1, hidden_Layer_size[0], 0);
    for(int i=1; i < NUM_HIDDEN_LAYER; i++){
        W[i] = makeMatrix(hidden_Layer_size[i-1], hidden_Layer_size[i], 0);//W[hidden] alloc
        B[i] = makeMatrix(1, hidden_Layer_size[i], 0);
    }
    W[NUM_HIDDEN_LAYER] = makeMatrix(hidden_Layer_size[NUM_HIDDEN_LAYER-1],output_Layer, 0);//W[n-1] alloc
    B[NUM_HIDDEN_LAYER] = makeMatrix(1, output_Layer, 0);

///////////////////// WEIGHT BIAS INITIALIZATION /////////////////////////////

    //Weights dimension check
    printf("=====W dimension=====\n");
    for(int i=0; i < NUM_HIDDEN_LAYER+1;i++){
        dummyMatrix(W[i]);dummyMatrix(B[i]);//가중치 추가
        printf("W[%d] : ",i);
        infoMatrix(W[i]);
        printf("B[%d] : ",i);
        infoMatrix(B[i]);
    }
    printf("=====================\n");
    






//////////////////////////////DEVICE MEMORY ALLOCATION & INITIALIZATION/////////////////////////////////////

    // 여기서는 디바이스 메모리 할당 및 초기화 한다. copy메모리로 메모리를 받아야 하기 때문에 앞에서 한번에 못한다.
    
    Matrix *dInput = moveMatrix(input, 1);
    Matrix *dlabel = moveMatrix(label, 1);

    Matrix *dW[NUM_HIDDEN_LAYER + 1];//W
    Matrix *dB[NUM_HIDDEN_LAYER + 1];//bias
    
    Matrix *dA[NUM_HIDDEN_LAYER + 1];//activation function layer
    Matrix *dO;//last softmax layer
    Matrix *dY;

    Matrix *dSigma[NUM_HIDDEN_LAYER + 1];//역전파할 때의 dA미분값, 차원은 dA와 같다.
    Matrix *dW_deriv[NUM_HIDDEN_LAYER + 1];//역전파 할떄의 dW미분값, 차원은 dW와 같다. 
    Matrix *dB_deriv[NUM_HIDDEN_LAYER + 1];//역전파 할 때의 dB미분값, 차원은 dB와 같다.

    Matrix *tmp;//그냥 일단 가지고 있자. 


///////////////////////////////dW & dB memory copy[0~3]//////////////////////////////////

    for(int i=0; i <= NUM_HIDDEN_LAYER; i++){
        dW[i] = copyMatrix(W[i], 1);// 이거 move로 바꿔도 되는 거 아닌가????? 일단은 넘어가자. 근데 필요하진 않을듯
        dB[i] = copyMatrix(B[i], 1);// 이거도 마찬가지
    }

    //dA MATRIX ALLOC
    for(int i=0; i < NUM_HIDDEN_LAYER; i++){
        dA[i] = makeMatrix(batch_size, hidden_Layer_size[i], 1);//b[hidden] alloc
    }
    dA[NUM_HIDDEN_LAYER] = makeMatrix(batch_size, output_Layer, 1);

    //dO MATRIX ALLOC
    dO = makeMatrix(batch_size, output_Layer, 1);


    //BACKPROP DERIVATIVE MATRIX ALLOCATION
    dW_deriv[0] = makeMatrix(input_Layer, hidden_Layer_size[0], 1);//W[0] alloc
    dB_deriv[0] = makeMatrix(1, hidden_Layer_size[0], 1);
    dSigma[0] = makeMatrix(batch_size, hidden_Layer_size[0], 1);
    for(int i=1; i < NUM_HIDDEN_LAYER; i++){
        dW_deriv[i] = makeMatrix(hidden_Layer_size[i-1], hidden_Layer_size[i], 1);//W[hidden] alloc
        dB_deriv[i] = makeMatrix(1, hidden_Layer_size[i], 1);
        dSigma[i] = makeMatrix(batch_size, hidden_Layer_size[i], 1);
    }
    dW_deriv[NUM_HIDDEN_LAYER] = makeMatrix(hidden_Layer_size[NUM_HIDDEN_LAYER-1],output_Layer, 1);
    dB_deriv[NUM_HIDDEN_LAYER] = makeMatrix(1, output_Layer, 1);
    dSigma[NUM_HIDDEN_LAYER] = makeMatrix(batch_size, output_Layer, 1);

    printf("=====dW=====\n");
    for(int i=0; i< sizeof(dW)/sizeof(Matrix*); i++){
        infoMatrix(dW[i]);
    }
    printf("=====dB=====\n");
    for(int i=0; i< sizeof(dB)/sizeof(Matrix*); i++){
        infoMatrix(dB[i]);
    }
    printf("=====dA=====\n");
    for(int i=0; i< sizeof(dA)/sizeof(Matrix*); i++){
        infoMatrix(dA[i]);
    }

/////////////////////////////DATA TRANSFER FROM FILE/////////////////////////////////
    
    
    
    dummyMatrix(input);//지금은 더미데이터지만 파일에서 가져와야한다. 




////////////////////////////////=FORWARD PASS=/////////////////////////////////
    
//     //나중에 메모리를 해제하는 것은 dA만으로 충분하다.
//     dA[0] = ReLU_inline(matmul_Bias_inline(dA[0], dInput, dW[0], dB[0]));

//     for(int i=1; i < NUM_HIDDEN_LAYER; i++){
//         dA[i] = ReLU_inline(matmul_Bias_inline(dA[i], dA[i-1], dW[i], dB[i]));//여기 계속 make하네 꼭 iteration마다 해제 해줘야함.
//     }


//     dA[NUM_HIDDEN_LAYER] = matmul_Bias_inline(dA[NUM_HIDDEN_LAYER], dA[NUM_HIDDEN_LAYER-1], dW[NUM_HIDDEN_LAYER], dB[NUM_HIDDEN_LAYER]);//마지막 레이어는 activation 을 통과하면 안됨.
    
//     //softMax 함수
//     dO = softMax_Rowwise_inline(dO, dA[NUM_HIDDEN_LAYER]);

//     ////////////////////////////////=LOSS CALCULATION=/////////////////////////////////

//     printf("dO:\n");
//     O = copyMatrix(dO, 0);
//     printMatrix(O);

//     float loss = 0;
//     for(int i=0; i < O->row; i++){//batch
//         printf("%d\n",(int)label->M[i]);
//         loss -= log(O->M[i * O->col + (int)label->M[i]]);
//     }
//     printf("loss : %f\n", loss);
//     //////////////////////////////////=BACKWARD PASS=/////////////////////////////////
//     //O_i-Y_i

//     dY = copyMatrix(label, 1);
//     matSub(dSigma[NUM_HIDDEN_LAYER-1], dO, dY);
    


    return 0;
}