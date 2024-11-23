#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "./Easy_Tensor/easy_tensor.h"

Tensor* dummyTensor(Tensor *ten){
    for(int i=0; i < ten->dim[0] * ten->stride[0]; i++){
        ten->T[i] = i;
    }
    return ten;
}


Tensor* dummyTensor2(Tensor *ten){
    for(int i=0; i < ten->sizeTensor; i++){
        ten->T[i] = 0;
    }
    for(int i=0; i < ten->dim[0]; i++){
        ten->T[i*ten->stride[0] + i]= 1;
    }
    return ten;
}
Tensor* dummyTensor3(Tensor *ten){
    for(int i=0; i < ten->sizeTensor; i++){
        ten->T[i] = 0;
    }
    for(int i=0; i < ten->dim[0]; i++){
        for(int j=0; j < ten->dim[1]; j++){
            ten->T[i*ten->stride[0] + j*ten->stride[1] + j]= 1;
        }
    }
    return ten;
}

#define CUDA_CHECK(status) \
    if (status != cudaSuccess) { \
        printf("CUDA Error: %s\n", cudaGetErrorString(status)); \
        return NULL; \
    }

#define CUBLAS_CHECK(status) \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS Error\n"); \
        return NULL; \
    }

// Function to perform batched matrix multiplication with broadcasting
Tensor* matmul_cublas_batched(Tensor* dC, Tensor* dA, Tensor* dB) {
    if (!dC || !dA || !dB) {
        printf("matmul_cublas_batched: One of the input tensors is NULL.\n");
        return NULL;
    }

    if (dC->device_type != dA->device_type || dA->device_type != dB->device_type) {
        printf("matmul_cublas_batched: Tensors are on different devices.\n");
        return NULL;
    }

    if (dA->num_dim < 2 || dB->num_dim < 2) {
        printf("matmul_cublas_batched: Input tensors must have at least 2 dimensions.\n");
        return NULL;
    }

    cudaSetDevice(dA->device_type - 1);
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Determine batch size
    int batchCount = (dC->num_dim >= 3) ? dC->dim[0] : 1;

    //M, K, N
    int M = dA->dim[dA->num_dim - 2];  // Rows of A
    int K = dA->dim[dA->num_dim - 1];  // Columns of A
    int N = dB->dim[dB->num_dim - 1];
    char batch_flag = 0;
    //make batch
    if(dC->num_dim == 3){
        if (dA->num_dim == 3 && dB->num_dim == 2) {
            // B needs to be broadcasted
            // Create batched version of B
            int B_b_dim[] = {dA->dim[0], dB->dim[0], dB->dim[1]};

            Tensor* B_batched = mallocTensor(B_b_dim, 3, dB->device_type);
            // Copy B into each batch
            for (int i = 0; i < batchCount; ++i) {
                CUDA_CHECK(cudaMemcpy(B_batched->T + i * B_batched->stride[0],
                                    dB->T,
                                    dB->sizeTensor * sizeof(float),
                                    cudaMemcpyDeviceToDevice));
            }
            dB = B_batched;
            batch_flag = 1;
        } else if (dB->num_dim == 3 && dA->num_dim == 2) {
            int A_b_dim[] = {dB->dim[0], dA->dim[0], dA->dim[1]};
            Tensor* A_batched = mallocTensor(A_b_dim, 3, dA->device_type);
            for (int i = 0; i < batchCount; ++i) {
                CUDA_CHECK(cudaMemcpy(A_batched->T + i * A_batched->stride[0],
                                    dA->T,
                                    dA->sizeTensor * sizeof(float),
                                    cudaMemcpyDeviceToDevice));
            }
            dA = A_batched;
            batch_flag = 2;
        } else if(dA->num_dim==dC->num_dim && dA->num_dim == dB->num_dim && dA->dim[0] == dB->dim[0]){
            //
        } 
        else {
            printf("matmul_cublas_batched: one of A, B must have at least 3 dimensions.\n");
            return NULL;
        }
    }
    

    // Now dA and dB both have batchCount batches
    // Leading dimensions
    int lda = K;
    int ldb = N;
    int ldc = N;

    // Strides
    long long strideA = (dA->num_dim >= 3) ? dA->stride[0] : M * K;
    long long strideB = (dB->num_dim >= 3) ? dB->stride[0] : K * N;
    long long strideC = (dC->num_dim >= 3) ? dC->stride[0] : M * N;

    float alpha = 1.0f;
    float beta = 0.0f;

    printf("M = %d, N = %d, K = %d\n", M, N, K);
    printf("lda = %d, ldb = %d, ldc = %d\n", lda, ldb, ldc);
    printf("strideA = %lld, strideB = %lld, strideC = %lld\n", strideA, strideB, strideC);
    printf("batchCount = %d\n", batchCount);

    // Perform batched matrix multiplication
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,  // Note: Swap M and N due to row-major storage
        &alpha,
        dB->T, ldb, strideB,
        dA->T, lda, strideA,
        &beta,
        dC->T, ldc, strideC,
        batchCount
    ));
    //clean
    if(batch_flag == 1){//B is batched
        freeTensor(dB);
    }else if(batch_flag == 2){//A is batched
        freeTensor(dA);
    }

    cublasDestroy(handle);

    return dC;
}

int main() {
    // Create dummy tensors
    Tensor* A = dummyTensor(makeTensor("3 196 768", 0));    // Batch of 3 matrices, each of size 4x3
    Tensor* dA = copyTensor(makeTensorbyShape(A, 1), A);
    Tensor* B = dummyTensor(makeTensor("768 2304", 0));      // Matrix of size 3x5
    Tensor* dB = copyTensor(makeTensorbyShape(B, 1), B);
    // printTensor(A);
    // printTensor(B);
    Tensor* C = makeTensor("3 196 2304", 0);                 // Batch of 3 matrices, each of size 4x5
    Tensor* dC = copyTensor(makeTensorbyShape(C, 1), C);

    // Perform batched matrix multiplication
    matmul_cublas_batched(dC, dA, dB);

    // Copy the result back to host and print
    freeTensor(printTensor(makeSubTensor(copyTensor(C, dC), "0 0 0", "8 8")));

    // Optionally compare with your own matmul function
    matmul(dC, dA, dB);
    freeTensor(printTensor(makeSubTensor(copyTensor(C, dC), "0 0 0", "8 8")));

    // Clean up
    freeTensor(A);
    freeTensor(B);
    freeTensor(C);
    freeTensor(dA);
    freeTensor(dB);
    freeTensor(dC);

    return 0;
}
