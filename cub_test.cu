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


// #define CUDA_CHECK(status) \
//     if (status != cudaSuccess) { \
//         printf("CUDA Error: %s\n", cudaGetErrorString(status)); \
//         return EXIT_FAILURE; \
//     }

// #define CUBLAS_CHECK(status) \
//     if (status != CUBLAS_STATUS_SUCCESS) { \
//         printf("cuBLAS Error\n"); \
//         return EXIT_FAILURE; \
//     }

// // Matrix multiplication for Tensors: C = alpha * A * B + beta * C
// int tensorGemm(cublasHandle_t handle, Tensor *A, Tensor *B, Tensor *C, float alpha, float beta) {
//     if (A->num_dim < 2 || B->num_dim < 2 || C->num_dim < 2) {
//         printf("Invalid tensor dimensions for matrix multiplication.\n");
//         return EXIT_FAILURE;
//     }

//     // Extract matrix dimensions
//     int M = C->dim[C->num_dim - 2];  // Rows of C
//     int N = C->dim[C->num_dim - 1];  // Columns of C
//     int K = A->dim[A->num_dim - 1];  // Columns of A and rows of B

//     // Leading dimensions (strides)
//     int lda = A->stride[A->num_dim - 2];
//     int ldb = B->stride[B->num_dim - 2];
//     int ldc = C->stride[C->num_dim - 2];

//     // Check tensor location (GPU)
//     if (A->device_type != C->device_type || B->device_type != C->device_type) {
//         printf("Tensors must be on the same device.\n");
//         return EXIT_FAILURE;
//     }

//     if (A->device_type != 1) { // Assuming 1 indicates GPU
//         printf("Tensors must be on the GPU for cuBLAS.\n");
//         return EXIT_FAILURE;
//     }

//     // Perform matrix multiplication using cuBLAS
//     CUBLAS_CHECK(cublasSgemm(
//         handle,
//         CUBLAS_OP_N, CUBLAS_OP_N,
//         N, M, K,
//         &alpha,
//         B->T, ldb,  // Matrix B, leading dimension
//         A->T, lda,  // Matrix A, leading dimension
//         &beta,
//         C->T, ldc   // Matrix C, leading dimension
//     ));

//     return EXIT_SUCCESS;
// }

// int main() {
//     cublasHandle_t handle;
//     CUBLAS_CHECK(cublasCreate(&handle));

//     // Example tensor setup (simplified)
//     int M = 1024, N = 1024, K = 1024;

//     // Allocate device memory for tensors (example)
//     Tensor* A = dummyTensor(makeTensor("5 4",0));
//     Tensor* dA = copyTensor(makeTensorbyShape(A, 1), A);
//     Tensor* dA_sub = makeSubTensor(dA, "1 0", "3 4");
//     Tensor*B = dummyTensor(makeTensor("4 5", 0));
//     Tensor*dB = copyTensor(makeTensorbyShape(B, 1), B);
//     Tensor*C = dummyTensor(makeTensor("4 5", 0));
//     Tensor*dC = copyTensor(makeTensorbyShape(C, 1), C);
//     // cudaMalloc((void **)&A.T, M * K * sizeof(float));
//     // cudaMalloc((void **)&B.T, K * N * sizeof(float));
//     // cudaMalloc((void **)&C.T, M * N * sizeof(float));

//     // // Set tensor dimensions
//     // A.dim = new int[2]{M, K};
//     // A.stride = new int[2]{K, 1};
//     // A.num_dim = 2;
//     // A.device_type = 1;

//     // B.dim = new int[2]{K, N};
//     // B.stride = new int[2]{N, 1};
//     // B.num_dim = 2;
//     // B.device_type = 1;

//     // C.dim = new int[2]{M, N};
//     // C.stride = new int[2]{N, 1};
//     // C.num_dim = 2;
//     // C.device_type = 1;

//     float alpha = 1.0f;
//     float beta = 0.0f;

//     // Perform the matrix multiplication
//     if (tensorGemm(handle, dA_sub, dB, dC, alpha, beta) != EXIT_SUCCESS) {
//         printf("Matrix multiplication failed.\n");
//         return EXIT_FAILURE;
//     }

//     printTensor(copyTensor(C, dC));

//     // Clean up
//     freeTensor(A);
//     freeTensor(B);
//     freeTensor(C);
//     freeTensor(dA);
//     freeTensor(dB);
//     freeTensor(dC);
//     cublasDestroy(handle);

//     printf("Matrix multiplication completed successfully.\n");
//     return 0;
// }

// #include <cublas_v2.h>
// #include <cuda_runtime.h>
// #include <stdio.h>
// #include "./Easy_Tensor/easy_tensor.h"

// Tensor* dummyTensor(Tensor *ten) {
//     for (int i = 0; i < ten->dim[0] * ten->stride[0]; i++) {
//         ten->T[i] = i;
//     }
//     return ten;
// }

// Tensor* dummyTensor2(Tensor *ten) {
//     for (int i = 0; i < ten->dim[0] * ten->stride[0]; i++) {
//         ten->T[i] = i % 2;
//     }
//     return ten;
// }

// #define CUDA_CHECK(status) \
//     if (status != cudaSuccess) { \
//         printf("CUDA Error: %s\n", cudaGetErrorString(status)); \
//         return EXIT_FAILURE; \
//     }

// #define CUBLAS_CHECK(status) \
//     if (status != CUBLAS_STATUS_SUCCESS) { \
//         printf("cuBLAS Error\n"); \
//         return EXIT_FAILURE; \
//     }

// // Matrix multiplication for batched Tensors: C = alpha * A * B + beta * C
// int tensorBatchedGemm(cublasHandle_t handle, Tensor *A, Tensor *B, Tensor *C, float alpha, float beta) {
//     if (A->num_dim != 3 || B->num_dim != 2 || C->num_dim != 3) {
//         printf("Invalid tensor dimensions for batched matrix multiplication.\n");
//         return EXIT_FAILURE;
//     }

//     // Extract matrix dimensions
//     int batch_count = A->dim[0];  // Number of matrices (4 in this case)
//     int M = C->dim[1];            // Rows of C (5)
//     int N = C->dim[2];            // Columns of C (5)
//     int K = A->dim[2];            // Columns of A and rows of B (5)

//     // Leading dimensions (strides)
//     int lda = A->stride[1];
//     int ldb = B->stride[0];
//     int ldc = C->stride[1];

//     // Strides between consecutive matrices in the batch
//     long long strideA = A->stride[0];
//     long long strideB = 0;  // Matrix B is the same for all batches
//     long long strideC = C->stride[0];

//     // Perform batched matrix multiplication using cuBLAS
//     CUBLAS_CHECK(cublasSgemmStridedBatched(
//         handle,
//         CUBLAS_OP_N, CUBLAS_OP_N,
//         N, M, K,
//         &alpha,
//         B->T, ldb, strideB,  // Matrix B, leading dimension, and stride
//         A->T, lda, strideA,  // Matrix A, leading dimension, and stride
//         &beta,
//         C->T, ldc, strideC,  // Matrix C, leading dimension, and stride
//         batch_count          // Number of matrices in the batch
//     ));

//     return EXIT_SUCCESS;
// }

// int main() {
//     cublasHandle_t handle;
//     CUBLAS_CHECK(cublasCreate(&handle));

//     // Create dummy tensors
//     Tensor* A = dummyTensor(makeTensor("4 5 5", 0));    // [4, 5, 5]
//     Tensor* dA = copyTensor(makeTensorbyShape(A, 1), A);
//     Tensor* B = dummyTensor2(makeTensor("5 5", 0));     // [5, 5]
//     Tensor* dB = copyTensor(makeTensorbyShape(B, 1), B);
//     Tensor* C = dummyTensor(makeTensor("4 5 5", 0));    // [4, 5, 5]
//     Tensor* dC = copyTensor(makeTensorbyShape(C, 1), C);

//     float alpha = 1.0f;
//     float beta = 0.0f;

//     // Perform batched matrix multiplication
//     if (tensorBatchedGemm(handle, dA, dB, dC, alpha, beta) != EXIT_SUCCESS) {
//         printf("Batched matrix multiplication failed.\n");
//         return EXIT_FAILURE;
//     }

//     // Copy the result back to host and print
//     printTensor(copyTensor(C, dC));

//     matmul(dC, dA, dB);


//     printTensor(copyTensor(C, dC));
//     // Clean up
//     freeTensor(A);
//     freeTensor(B);
//     freeTensor(C);
//     freeTensor(dA);
//     freeTensor(dB);
//     freeTensor(dC);
//     cublasDestroy(handle);

//     printf("Batched matrix multiplication completed successfully.\n");
//     return 0;
// }


// // Error-checking macros
// #define CUDA_CHECK(status) \
//     if (status != cudaSuccess) { \
//         printf("CUDA Error: %s\n", cudaGetErrorString(status)); \
//         return NULL; \
//     }

// #define CUBLAS_CHECK(status) \
//     if (status != CUBLAS_STATUS_SUCCESS) { \
//         printf("cuBLAS Error\n"); \
//         return NULL; \
//     }

// // Helper function to calculate batch size from higher dimensions
// int calculateBatchSize(int *dims, int num_dim) {
//     int batch_size = 1;
//     for (int i = 0; i < num_dim - 2; i++) {
//         batch_size *= dims[i];
//     }
//     return batch_size;
// }
// Tensor* matmul_cublas(Tensor* C, Tensor* A, Tensor* B){
//         cublasSgemmStridedBatched(
//         handle,
//         CUBLAS_OP_N, CUBLAS_OP_N,
//         N, M, K,
//         &alpha,
//         dB, N, K * N,
//         dA, K, M * K,
//         &beta,
//         dC, N, M * N,
//         batchCount
//     );
//     return C;
// }
// #include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "./Easy_Tensor/easy_tensor.h"

// #define CUDA_CHECK(status) \
//     if (status != cudaSuccess) { \
//         printf("CUDA Error: %s\n", cudaGetErrorString(status)); \
//         return NULL; \
//     }

// #define CUBLAS_CHECK(status) \
//     if (status != CUBLAS_STATUS_SUCCESS) { \
//         printf("cuBLAS Error\n"); \
//         return NULL; \
//     }

// Tensor* matmul_cublas(Tensor* dC, Tensor* dA, Tensor* dB) {
//     if (!dC || !dA || !dB) {
//         printf("matmul_cublas: One of the input tensors is NULL.\n");
//         return NULL;
//     }

//     if (dC->device_type != dA->device_type || dA->device_type != dB->device_type) {
//         printf("matmul_cublas: Tensors are on different devices.\n");
//         return NULL;
//     }

//     if (dC->num_dim < 2 || dA->num_dim < 2 || dB->num_dim < 2) {
//         printf("matmul_cublas: Tensors must have at least 2 dimensions.\n");
//         return NULL;
//     }

//     cudaSetDevice(dA->device_type - 1);
//     cublasHandle_t handle;
//     CUBLAS_CHECK(cublasCreate(&handle));

 
//     // Matrix dimensions
//     int M = dA->dim[dA->num_dim - 2];  // Rows of C and A
//     int N = dB->dim[dB->num_dim - 1];  // Columns of C and B
//     int K = dA->dim[dA->num_dim - 1];  // Columns of A and rows of B

//     // Batch size
//     int batchCount = (dC->num_dim >= 3) ? dC->dim[dC->num_dim - 3] : 1;

//     // Leading dimensions
//     //[3 5 4] [4 5]
//     int lda = M;
//     int ldb = K;
//     int ldc = M;

//     // Strides
//     long long strideA = (dA->num_dim >= 3) ? dA->stride[0] : 0;
//     long long strideB = (dB->num_dim >= 3) ? dB->stride[0] : 0;  // Matrix B is the same for all batches
//     long long strideC = (dC->num_dim >= 3) ? dC->stride[0] : 0;
//     float alpha = 1.0f;
//     float beta = 0.0f;

//     printf("M = %d, N = %d, K = %d\n", M, N, K);
//     printf("lda = %d, ldb = %d, ldc = %d\n", lda, ldb, ldc);
//     printf("strideA = %lld, strideB = %lld, strideC = %lld\n", strideA, strideB, strideC);
//     printf("batchCount = %d\n", batchCount);

//     // Corrected transpose parameters
//     CUBLAS_CHECK(cublasSgemmStridedBatched(
//         handle,
//         CUBLAS_OP_N, CUBLAS_OP_N,  // No transposition
//         M, N, K,
//         &alpha,
//         dA->T, lda, strideA,       // Matrix A
//         dB->T, ldb, strideB,       // Matrix B
//         &beta,
//         dC->T, ldc, strideC,       // Matrix C
//         batchCount
//     ));

//     cublasDestroy(handle);
//     return dC;
// }

// Tensor* matmul_cublas(Tensor* dC, Tensor* dA, Tensor* dB) {
//     if (!dC || !dA || !dB) {
//         printf("matmul_cublas: One of the input tensors is NULL.\n");
//         return NULL;
//     }

//     if (dC->device_type != dA->device_type || dA->device_type != dB->device_type) {
//         printf("matmul_cublas: Tensors are on different devices.\n");
//         return NULL;
//     }

//     if (dC->num_dim < 2 || dA->num_dim < 2 || dB->num_dim < 2) {
//         printf("matmul_cublas: Tensors must have at least 2 dimensions.\n");
//         return NULL;
//     }

//     cudaSetDevice(dA->device_type - 1);
//     cublasHandle_t handle;
//     CUBLAS_CHECK(cublasCreate(&handle));

//     // Matrix dimensions
//     int M = dA->dim[dA->num_dim - 2];  // Rows of A
//     int K = dA->dim[dA->num_dim - 1];  // Columns of A
//     int N = dB->dim[dB->num_dim - 2];  // Rows of B (since we will transpose B)

//     // Batch size
//     int batchCount = (dA->num_dim >= 3) ? dA->dim[dA->num_dim - 3] : 1;

//     // Leading dimensions
//     int lda = K;  // For CUBLAS_OP_N, lda >= max(1, M)
//     int ldb = N;  // For CUBLAS_OP_T, ldb >= max(1, N)
//     int ldc = N;  // ldc >= max(1, M)

//     // Strides
//     long long strideA = (dA->num_dim >= 3) ? dA->stride[0] : M * K;
//     long long strideB = 0;  // Assuming B is the same for all batches
//     long long strideC = (dC->num_dim >= 3) ? dC->stride[0] : M * N;

//     float alpha = 1.0f;
//     float beta = 0.0f;

//     printf("M = %d, N = %d, K = %d\n", M, N, K);
//     printf("lda = %d, ldb = %d, ldc = %d\n", lda, ldb, ldc);
//     printf("strideA = %lld, strideB = %lld, strideC = %lld\n", strideA, strideB, strideC);
//     printf("batchCount = %d\n", batchCount);

//     // Corrected function call
//     CUBLAS_CHECK(cublasSgemmStridedBatched(
//         handle,
//         CUBLAS_OP_N, CUBLAS_OP_T,  // No transpose for A, transpose B
//         N, M, K,
//         &alpha,
//         dB->T, ldb, strideB,       // Matrix B
//         dA->T, lda, strideA,       // Matrix A
//         &beta,
//         dC->T, ldc, strideC,       // Matrix C
//         batchCount
//     ));

//     cublasDestroy(handle);
//     return dC;
// }



int main(){
        // Create dummy tensors
    Tensor* A = dummyTensor(makeTensor("3 5 5", 0));    // [4, 5, 5]
    Tensor* dA = copyTensor(makeTensorbyShape(A, 1), A);
    Tensor* dAt = makeTensor("3 5 5", 1);
    Tensor* B = dummyTensor(makeTensor("5 5", 0));     // [5, 5]
    Tensor* dB = copyTensor(makeTensorbyShape(B, 1), B);
    Tensor* dBt = makeTensorbyShape(B, 1);
    Tensor* bias = dummyTensor(makeTensor("5", 0));
    Tensor* dbias = copyTensor(makeTensorbyShape(bias, 1), bias);
    printTensor(A);
    printTensor(B);
    // add_Bias(dA, dB);
    printTensor(copyTensor(A, dA));
    Tensor* C = makeTensor("3 5 5", 0);    // [4, 5, 5]
    Tensor* dC = copyTensor(makeTensorbyShape(C, 1), C);


    // Perform batched matrix multiplication

    // Copy the result back to host and print

    matmul_cublas_batched_bias(dC, dA, dB, dbias);
    printTensor(copyTensor(C, dC));

    matmul_bias(dC, dA, dB, dbias, 0);
    printTensor(copyTensor(C, dC));

    // Clean up
    freeTensor(A);
    freeTensor(B);
    freeTensor(C);
    freeTensor(dA);
    freeTensor(dB);
    freeTensor(dC);
}