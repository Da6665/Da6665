#include <stdio.h>
#include <cuda_runtime.h>

__global__ void test() {
    printf("Hello CUDA! threadIdx.x = %d\n", threadIdx.x);
}

int main() {
    test<<<1, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
