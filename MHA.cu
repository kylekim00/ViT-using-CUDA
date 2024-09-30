#include<stdio.h>
#include<stdlib.h>
#include<float.h>
#include <curand.h>
#include <curand_kernel.h>
#include "./Easy_Tensor/easy_tensor.h"

#include "MHA.h"
#include<string.h>


//그저 flashAttention에 최적화된 형태로 만드는 것이다. 지금은.


//BLOCK
//QKV(768 2304) PROJ(768 768) MLP(768 3072) MLP(3072 768)
#define NUM_OF_TENSORS_IN_MHABLOCK 12
#define WQKV_INX 2

Tensor** makeMHABlock(int device_type){
    Tensor** newBlock = (Tensor**)malloc(sizeof(Tensor*) * NUM_OF_TENSORS_IN_MHABLOCK);

    //norm
    newBlock[0] = makeTensor("768", device_type);       // NORM1 weight
    newBlock[1] = makeTensor("768", device_type);       // NORM1 bias

    newBlock[2] = makeTensor("768 2304", device_type);  // QKV(768 2304)
    newBlock[3] = makeTensor("2304", device_type);       // QKV bias

    newBlock[4] = makeTensor("768 768", device_type);   // Linear Projection after concat (768 768)
    newBlock[5] = makeTensor("768", device_type);       // Proj bias

    newBlock[6] = makeTensor("768", device_type);       // NORM2 weight
    newBlock[7] = makeTensor("768", device_type);       // NORM2 bias

    newBlock[8] = makeTensor("768 3072", device_type);  // MLP1 (768 3072)
    newBlock[9] = makeTensor("3072", device_type);      // MLP1 bias

    newBlock[10] = makeTensor("3072 768", device_type);  // MLP2 (3072 768)
    newBlock[11] = makeTensor("768", device_type);       // MLP2 bias

    return newBlock;
}



void freeMHABlock(Tensor** block){
    for(int i=0; i < NUM_OF_TENSORS_IN_MHABLOCK; i++)
        freeTensor(block[i]);
    free(block);
}


//////////////////////////////////////////////////////////////////////////////////////////////////
Tensor** copyMHABlockfromFILE(Tensor** block, const char* file_name){
    
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

    size_t num_elements = fread(block[0]->T, sizeof(float), block[0]->sizeTensor, file);
    if (num_elements != block[0]->sizeTensor) {
        printf("Error reading file\n");
        return NULL;
    }

    for(int i=1; i < NUM_OF_TENSORS_IN_MHABLOCK; i++){
    
        num_elements = fread(block[i]->T, sizeof(float), block[i]->sizeTensor, file);
        if (num_elements != block[i]->sizeTensor) {
            printf("Error reading file\n");
            return NULL;
        }
    }

    fclose(file);

    return block;
}

Tensor** copyMHABlock(Tensor** dst, Tensor** src){
    if(dst == NULL || src == NULL){
        printf("no block.\n");
        return NULL;
    }
    for(int i=0; i < NUM_OF_TENSORS_IN_MHABLOCK; i++){
        copyTensor(dst[i], src[i]);
    }
    return dst;
}

Tensor* copyTensorfromFILE(Tensor* dst, const char* file_name){
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

//for FLASH ATTENTION
#define HIDDEN_DIM 64 // QKV = head * hidden dim * 3
#define ATTN_TILE_SIZE 8 //GPU 블럭 크기

// 일단 8로 tile을 맞춰준다고 생각하고 진행한다. 768%8=0, 2304%8=0

__global__ void flashAttention_MHA_(float *Out, float *QKV, int *dQKV_dim){// dQKV 4 196 2304 
    __shared__ float Q[ATTN_TILE_SIZE][HIDDEN_DIM];
    // __shared__ float KV[ATTN_TILE_SIZE * HIDDEN_DIM];
    __shared__ float KV[ATTN_TILE_SIZE][HIDDEN_DIM];
    __shared__ float O[ATTN_TILE_SIZE][HIDDEN_DIM];

    __shared__ float SP[ATTN_TILE_SIZE][ATTN_TILE_SIZE];
    __shared__ float l[ATTN_TILE_SIZE];
    __shared__ float m[ATTN_TILE_SIZE];
    // __shared__ float tmp_max[ATTN_TILE_SIZE];
    

    //shared mem initialization.
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int num_of_head = dQKV_dim[2] / (3 * HIDDEN_DIM);
    //Z는 batch 와 head의 inx를 가지고 있다.
    int z_batch = blockIdx.z / num_of_head; //batch_Idx
    int z_head = blockIdx.z % num_of_head;  //head_Idx
    
    for(int i=0; i < HIDDEN_DIM ;i+= ATTN_TILE_SIZE){
        O[threadIdx.y][threadIdx.x + i] = 0;
    }

    m[threadIdx.y] = - __FLT_MAX__;
    l[threadIdx.y] = 0;
    __syncthreads();

    

    
    //if(row && col) 안할거임. 왜냐, 시간문제 걸리니까 일단 배수인 것으로 하고 할거임.
    //196이 8로 나누어 떨어지지 않기 때문에 해야할듯....
    
    //Q init
    if(row < dQKV_dim[1]){
        for(int i=0; i < HIDDEN_DIM ;i+= ATTN_TILE_SIZE){
            Q[threadIdx.y][i+threadIdx.x] = QKV[z_batch * dQKV_dim[3]/*matdim*/ + row * dQKV_dim[4]/*rowdim*/ + z_head * (HIDDEN_DIM*3/*headdim*/) + col + i];//여기 3은 QKV 3.
        }
    }else{
        for(int i=0; i < HIDDEN_DIM ;i+= ATTN_TILE_SIZE){
                Q[threadIdx.y][i+threadIdx.x] = 0;
        }
    }
    __syncthreads();


    float tmp;
    
    for(int iter=0; iter < dQKV_dim[1]; iter+= ATTN_TILE_SIZE){// PATCH_SIZE를 이제부터 쭉 돌거임.

        //load K
        if((iter + threadIdx.y) < dQKV_dim[1]){
            for(int i=0; i < HIDDEN_DIM; i+= ATTN_TILE_SIZE){
                KV[threadIdx.y][threadIdx.x+i] = QKV[z_batch * dQKV_dim[3]/*matdim*/ + (iter+threadIdx.y) * dQKV_dim[4]/*rowdim*/ + z_head * (HIDDEN_DIM * 3/*headdim*/) + HIDDEN_DIM/*k*/ + col + i];
            }
        }else{
            for(int i=0; i < HIDDEN_DIM; i+= ATTN_TILE_SIZE){
                KV[threadIdx.y][threadIdx.x+i] = 0;
            }
        }
        __syncthreads();
        
        //QK matmul
        tmp = 0;
        for(int i=0; i < HIDDEN_DIM; i++){
            tmp += Q[threadIdx.y][i] * KV[threadIdx.x][i];
        }
        ////////////////////////////////////////////////////////////////////////////////////
        //max값을 위한 접근을 위한 SP. 이거는 직접적인 ram부분 접근이 아니기 때문에 굳이 row를 막아줄 필요는 없다.
        SP[threadIdx.y][threadIdx.x] = tmp;
        __syncthreads();
        
        



        
        // rowMax(최신 max값을 tmp_max에 저장.)
        if(threadIdx.x== 0){//tmp_max[] -> KV[0][]//이거 만약 TILESIZE가 64보다 커지면 문제된다.
            KV[0][threadIdx.y] = m[threadIdx.y];//-
            for(int i=0; i < ATTN_TILE_SIZE; i++){
                if(iter + i < dQKV_dim[1]){/////////////////////////////현재 작업중
                    if(SP[threadIdx.y][i] > KV[0][threadIdx.y])//-
                        KV[0][threadIdx.y] = SP[threadIdx.y][i];//-
                }
            }
        }
        __syncthreads();

        


        //tmp 계속 QK값 담고있음.
        //물결P 구하기
        float r_max = (KV[0][threadIdx.y] > m[threadIdx.y])? KV[0][threadIdx.y] : m[threadIdx.y];//여기서부터 tmp_max는 필요 없게 된다.//-
        __syncthreads();

        if ((iter + threadIdx.x) < dQKV_dim[1]) {
            SP[threadIdx.y][threadIdx.x] = expf(tmp - r_max);
        } else {
            SP[threadIdx.y][threadIdx.x] = 0;
        }
        // SP[threadIdx.y][threadIdx.x] = expf(tmp - r_max);//rowsum을 구하기 위한 메모리
        __syncthreads();



        //////////////rowsum & sum 계산///////////////

        

        if(threadIdx.x == 0){
            float tmp_l = 0;
            for(int i=0; i < ATTN_TILE_SIZE; i++){
                if(iter + i < dQKV_dim[1])/////////////////////////////현재 작업중
                    tmp_l += SP[threadIdx.y][i];
            }

            l[threadIdx.y] = expf(m[threadIdx.y] - r_max) * l[threadIdx.y] + tmp_l;
        }
        __syncthreads();
        
        //load V
        if(iter + threadIdx.y < dQKV_dim[1]){
            for(int i=0; i < HIDDEN_DIM; i+= ATTN_TILE_SIZE){
                KV[threadIdx.y][threadIdx.x+i] = QKV[z_batch * dQKV_dim[3]/*matdim*/ + (iter+threadIdx.y) * dQKV_dim[4]/*rowdim*/ + z_head * (HIDDEN_DIM * 3/*headdim*/) + HIDDEN_DIM*2/*V*/ + col + i];
            }
        }else{
            for(int i=0; i < HIDDEN_DIM; i+= ATTN_TILE_SIZE){
                KV[threadIdx.y][threadIdx.x+i] = 0;
            }
        }
        __syncthreads();

        // if(iter==11*ATTN_TILE_SIZE && row==192 &&col==0 && blockIdx.z==0){
        //     printf("++++++++++++++++++++++++++++++++++++++++++++++\n");
        //     for(int i=0; i < ATTN_TILE_SIZE; i++){
        //         for(int j=0; j < ATTN_TILE_SIZE; j++){
        //             printf("%0.2f\t", SP[i][j]);
        //         }printf("\n");
        //     }
        // }
        // __syncthreads();
        // if(iter==11*ATTN_TILE_SIZE && row==192 &&col==0 && blockIdx.z==0){
        //     printf("++++++++++++++++++++++++++++++++++++++++++++++\n");
        //     for(int i=0; i < ATTN_TILE_SIZE; i++){
        //         for(int j=0; j < ATTN_TILE_SIZE; j++){
        //             printf("%0.2f\t", KV[i][j]);
        //         }printf("\n");
        //     }
        // }
        // __syncthreads();
        
//////////////////////////////////일단은 여기까지 확인//////////////////////////////////
        
        for(int i=0; i < HIDDEN_DIM; i+= ATTN_TILE_SIZE){
            //SVt matmul
            tmp = 0;

            for(int j=0; j < ATTN_TILE_SIZE; j++){
                tmp += SP[threadIdx.y][j] * KV[j][threadIdx.x + i];
            }
            __syncthreads();
            // if(iter==8 && col==0 && blockIdx.z==0){
            //     printf("<%d> : tmp: %f, %f\n", row, tmp, m[threadIdx.y] - r_max);
            // }
            //__syncthreads();

            //diag(l) O 곱해주기 exp는 최솟값일 때는 계산하지 않는다.
            if(m[threadIdx.y] != -__FLT_MAX__){
                tmp += O[threadIdx.y][threadIdx.x + i] * expf(m[threadIdx.y] - r_max);
            }else{
                tmp += O[threadIdx.y][threadIdx.x + i];
            }
            __syncthreads();


            O[threadIdx.y][threadIdx.x + i] = tmp;
            __syncthreads();
            ////메모리 체크/////
            // if(row ==192 && col == 0 && blockIdx.z==0 && iter ==11 * ATTN_TILE_SIZE && i==0){
            //     printf("++++++++++++++++++++++++++++++++++++++++++++++\n");
            //     for(int i=0; i < ATTN_TILE_SIZE; i++){
            //         for(int j=0; j < ATTN_TILE_SIZE; j++){
            //             printf("%0.2f\t", O[i][j]);
            //         }printf("\n");
            //     }

            //     // printf("<blockIdx:[%d %d]>O : %f\n",z_batch,z_head, tmp);
            // }
            __syncthreads();
        }

        
        // if(row==0 &&col==0 && blockIdx.z==0){
        //     printf("+++++++++++++++++++++++++++++++++++\n");
        //     for(int i=0; i< ATTN_TILE_SIZE;i++){
        //         printf("%f ", l[i]);
        //     }printf("\n");
        // }
        // __syncthreads();

        m[threadIdx.y] = r_max;
        __syncthreads();
    }
    //내보내기
    if(row < dQKV_dim[1]){
        for(int i=0; i < HIDDEN_DIM; i+= ATTN_TILE_SIZE){
            O[threadIdx.y][threadIdx.x + i] /= l[threadIdx.y];
            __syncthreads();
            
            // if(i==0 && row==192 &&col==0 && blockIdx.z==0){
            //     printf("++++++++++++++++++++++++++++++++++++++++++++++\n");
            //     for(int i=0; i < ATTN_TILE_SIZE; i++){
            //         for(int j=0; j < ATTN_TILE_SIZE; j++){
            //             printf("%0.2f\t", O[i][j]);
            //         }printf("\n");
            //     }
            //     printf("QKV : %d\n", dQKV_dim[1]);
            // }
            // __syncthreads();
            
            Out[z_batch * dQKV_dim[1] * HIDDEN_DIM * num_of_head/* O matdim*/ + row * HIDDEN_DIM * num_of_head/*rowdim*/ + (z_head*HIDDEN_DIM)/*여기에 head별로 또 나뉘어야함.*/+col + i] = O[threadIdx.y][threadIdx.x + i];
            __syncthreads();
        }
    }
}



Tensor* flashAttention_MHA(Tensor* O, Tensor* dQKV){
    if(!O||!dQKV){
        printf("no Tensor\n");
        return NULL;
    }
    if(O->device_type != dQKV->device_type){
        printf("Two matrices on different device.\n");
        return NULL;
    }

    cudaSetDevice(dQKV->device_type-1);
    dim3 dimGrid(1, (dQKV->dim[1] + ATTN_TILE_SIZE -1) / ATTN_TILE_SIZE, dQKV->dim[0] * dQKV->dim[2]/(3 * HIDDEN_DIM)); //dim은 4x3x32x32를 matmul하는 경우 12가 들어가게 된다.
    dim3 dimBlock(ATTN_TILE_SIZE, ATTN_TILE_SIZE);
    flashAttention_MHA_<<<dimGrid, dimBlock>>>(O->T, dQKV->T, dQKV->d_dim_stride);
    return O;
}