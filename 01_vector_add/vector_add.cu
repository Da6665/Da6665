#include <iostream>
#include <cuda_runtime.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

__global__ void elementwise_add(
    float* a,
    float* b,
    float* c,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1024;
    const size_t bytes = N * sizeof(float);

    // CPU 数据
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];

    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    // GPU 数据
    float *a, *b, *c;

    cudaMalloc(&a, bytes);
    cudaMalloc(&b, bytes);
    cudaMalloc(&c, bytes);

    cudaMemcpy(a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b, h_b, bytes, cudaMemcpyHostToDevice);

    // 启动 Kernel
    int block_size = 256;
    int grid_size = CEIL(N, block_size);

    elementwise_add<<<grid_size, block_size>>>(a, b, c, N);

    cudaMemcpy(h_c, c, bytes, cudaMemcpyDeviceToHost);

    // 检查前几个结果
    for (int i = 0; i < 10; i++) {
        std::cout << h_a[i] << " + "
                  << h_b[i] << " = "
                  << h_c[i] << std::endl;
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}