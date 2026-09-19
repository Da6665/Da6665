#include <iostream>
#include <cuda_runtime.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

// GPU Kernel
__global__ void elementwise_add_float4(
    float* a, float* b, float* c, int N
) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;

    if (idx + 3 < N) {
        float4 tmp_a = FLOAT4(a[idx]);
        float4 tmp_b = FLOAT4(b[idx]);

        float4 tmp_c;
        tmp_c.x = tmp_a.x + tmp_b.x;
        tmp_c.y = tmp_a.y + tmp_b.y;
        tmp_c.z = tmp_a.z + tmp_b.z;
        tmp_c.w = tmp_a.w + tmp_b.w;

        FLOAT4(c[idx]) = tmp_c;
    }
}

int main() {
    const int N = 1024;
    const int bytes = N * sizeof(float);

    float *a, *b, *c;

    cudaMalloc(&a, bytes);
    cudaMalloc(&b, bytes);
    cudaMalloc(&c, bytes);

    int block_size = 256;
    int grid_size = CEIL(CEIL(N, 4), block_size);

    elementwise_add_float4<<<grid_size, block_size>>>(a, b, c, N);

    cudaDeviceSynchronize();

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}