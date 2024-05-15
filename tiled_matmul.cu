#include <stdio.h>
#include <cuda_runtime.h>

#define tile_SIZE 8

// __global__ void MM_naive(int *A, int *B, int *C, int k){
//     int row = blockDim.y * blockIdx.y + threadIdx.y;
//     int col = blockDim.x * blockIdx.x + threadIdx.x;
//     C[row][col] = 0;
//     for(int i=0; i < k; i++){
//         C[row][col] += A[row][i] * B[i][col];
//     }
// }

__global__ void tiledMM(int *A, int *B, int *C, int M, int K){
    // row, col : 얘네들은 쓰레드의 위치를 나타내니까 쓰레드의 구분이 연산과정에서 필요할 때 써주면 된다.
    // threadIdx.y, threadIdx.x : 얘네들은 블럭 크기의 메모리를 다루어줄 때 쓰면된다. 차이점은 block을 구분지어야 할 필요가 있는 cuda 연산에서는 block을 붙여줘야하고
    // shared memory 와 같이 그냥 안에서 일어나는 것은  idx로 해줘도 된다는 것이다.
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ int s_a[tile_SIZE][tile_SIZE];
    __shared__ int s_b[tile_SIZE][tile_SIZE];

    int tmp = 0;

    for(int i=0; i < K; i += tile_SIZE){//i는 tile_size만큼 늘어난다.....
        //값을 가져올 때 A는 rowtile을 기준으로 C의 row를 기준으로 가져와야하고 B는 column tile을 기준으로 들고 와야하기 때문에 
        s_a[threadIdx.y][threadIdx.x] = A[row * K + (i + threadIdx.x)];
        s_b[threadIdx.y][threadIdx.x] = B[(i + threadIdx.y) * M + col]; // 먼저 i * N으로 타일에 포지션을 잡고, threadidx.y는 타일의 y 인덱스를 가리킨다. 
        __syncthreads();
        // printf("A: bidx : (%d,%d), tidx : (%d,%d) | %d\n",blockIdx.y,blockIdx.x,threadIdx.y,threadIdx.x, s_a[threadIdx.y][threadIdx.x]);
        // __syncthreads();
        // printf("B: bidx : (%d,%d), tidx : (%d,%d) | %d\n",blockIdx.y,blockIdx.x,threadIdx.y,threadIdx.x, s_b[threadIdx.y][threadIdx.x]);
        // 이제 계산한다. 
        for(int j=0; j < tile_SIZE; j++){
            tmp += s_a[threadIdx.y][j] * s_b[j][threadIdx.x];
        }

        __syncthreads();
    }
    // printf("(%d, %d) %d\n", row, col, tmp);
    C[row * M + col] = tmp;
}


__global__ void tiledMM_parai(){
    
}


void matmul(int *dA, int *dB, int *dC, int N, int M, int K){
    if(!(N % tile_SIZE) && !(M % tile_SIZE) && !(K % tile_SIZE)){
        dim3 dimGrid(M / tile_SIZE, N / tile_SIZE);//num of blocks per grid<threadidx x==t_size,threadidx.y==t_size>
        // printf("%d, %d", N/tile_SIZE, M/tile_SIZE);
        dim3 dimBlock(tile_SIZE, tile_SIZE);//num of threads per block<threadidx x==t_size,threadidx.y==t_size>
        tiledMM<<<dimGrid, dimBlock>>>(dA, dB, dC, M, K);
    }
}


int main(){
    int N = 16, K = 8, M = 8;
    int *A, *B, *C;
    size_t bytes_A = N * K * sizeof(int);
    size_t bytes_B = K * M * sizeof(int);
    size_t bytes_C = N * M * sizeof(int);

    A = (int*)malloc(bytes_A);
    B = (int*)malloc(bytes_B);
    C = (int*)malloc(bytes_C);

    int *dA, *dB, *dC;
    int tmp;
    cudaGetDevice(&tmp);
    cudaGetDeviceCount(&tmp);
    cudaMalloc(&dA, bytes_A);
    cudaMalloc(&dB, bytes_B);
    cudaMalloc(&dC, bytes_C);

    //init
    for(int i=0; i < N*K; i++){
        A[i] = i;
    }

    for(int i=0; i < K*M; i++){
        B[i] = i;
    }
    cudaMemcpy(dA, A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, bytes_B, cudaMemcpyHostToDevice);

    matmul(dA, dB, dC, N, M, K);
    cudaMemcpy(C, dC, bytes_C, cudaMemcpyDeviceToHost);
    for(int i=0; i < N; i++){
        for(int j=0; j < M; j++){
            printf("%d ", C[i*M + j]);
        }
        printf("\n");
    }
    // Free memory
    free(A);
    free(B);
    free(C);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
