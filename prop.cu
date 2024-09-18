#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Total global memory: %zu bytes\n", prop.totalGlobalMem);
        printf("  Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Registers per block: %d\n", prop.regsPerBlock);
        printf("  Warp size: %d\n", prop.warpSize);
        printf("  Memory pitch: %zu bytes\n", prop.memPitch);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max threads dimensions: x = %d, y = %d, z = %d\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max grid dimensions: x = %d, y = %d, z = %d\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Clock rate: %d KHz\n", prop.clockRate);
        printf("  Total constant memory: %zu bytes\n", prop.totalConstMem);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Texture alignment: %zu bytes\n", prop.textureAlignment);
        printf("  Device overlap: %d\n", prop.deviceOverlap);
        printf("  Multi-processor count: %d\n", prop.multiProcessorCount);
        printf("  Kernel execution timeout enabled: %d\n", prop.kernelExecTimeoutEnabled);
        printf("  Integrated: %d\n", prop.integrated);
        printf("  Can map host memory: %d\n", prop.canMapHostMemory);
        printf("  Compute mode: %d\n", prop.computeMode);
        printf("  Concurrent kernels: %d\n", prop.concurrentKernels);
        printf("  ECC enabled: %d\n", prop.ECCEnabled);
        printf("  PCI bus ID: %d\n", prop.pciBusID);
        printf("  PCI device ID: %d\n", prop.pciDeviceID);
        printf("  PCI domain ID: %d\n", prop.pciDomainID);
        printf("  TCC driver: %d\n", prop.tccDriver);
        printf("  Memory clock rate: %d KHz\n", prop.memoryClockRate);
        printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
        printf("  L2 cache size: %d bytes\n", prop.l2CacheSize);
        printf("  Max threads per multi-processor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Async engine count: %d\n", prop.asyncEngineCount);
        printf("  Unified addressing: %d\n", prop.unifiedAddressing);
        printf("  Peak memory bandwidth: %f GB/s\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  Concurrent copy and execution: %s\n", (prop.deviceOverlap ? "Yes" : "No"));
        printf("  Number of asynchronous engines: %d\n", prop.asyncEngineCount);
        printf("  Stream priorities supported: %d\n", prop.streamPrioritiesSupported);
        printf("  Global L1 cache supported: %d\n", prop.globalL1CacheSupported);
        printf("  Local L1 cache supported: %d\n", prop.localL1CacheSupported);
        printf("  Shared memory per multiprocessor: %zu bytes\n", prop.sharedMemPerMultiprocessor);
        printf("  Registers per multiprocessor: %d\n", prop.regsPerMultiprocessor);
        printf("  Managed memory: %d\n", prop.managedMemory);
        printf("  Is multi-GPU board: %d\n", prop.isMultiGpuBoard);
        printf("  Multi-GPU board group ID: %d\n", prop.multiGpuBoardGroupID);
        printf("\n");
    }
    return 0;
}

