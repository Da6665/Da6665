#include <stdio.h>
#include <stdlib.h>
#include "utils.cuh"


//朴素实现:合并读入
//input：M行N列   output:N行M列
__global__ void transpose_v0(float* input,float* output,int M,int N){
    const int row=blockDim.y*blockIdx.y+threadIdx.y;
    const int col=blockDim.x*blockIdx.x+threadIdx.x;

    if(row<M && col<N){
        output[col*M +row]=input[row * N+col];
    }
}

//合并写入
//相邻线程一般在同一行中
__global__ void transpose_v1(float* input,float* output,int M,int N){
    int row=blockDim.y*blockIdx.y+threadIdx.y;
    int col=blockDim.x*blockIdx.x+threadIdx.x;

    //row:output的行=input的列
    //col:output的列=input的行
    if(row<N && col<M){
        output[row*M+col]=input[col*N+row];
    }
}

//显示调用_ldg(),减少不合并读入的损失

__global__ void transpose_v2(float* input ,float* output,int M,int N){
    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N && col < M) {
        output[row * M + col] = __ldg(&input[col * N + row]);
    }
}

//使用共享内存中转， 合并读取+写入,对共享内存做padding,以避免 bank conflict
//bank conflict:
//英伟达芯片中一个block有32个blank
//bankid=(块内地址)/4  %32;
//对于 S[32][32],访问同一列的不同行元素，则会发生 bank conflict;

template<BLOCKSIZE,BLOCKSIZE>
__global__ void transpose_v3(float* input ,float* output,int M,int N){
    __shared__ float S[BLOCKSIZE][BLOCKSIZE+1];//+1做padding，避免bank conflict
    const int bx = blockIdx.x * TILE_DIM;
    const int by = blockIdx.y * TILE_DIM;
    const int x1 = bx + threadIdx.x;
    const int y1 = by + threadIdx.y;
    //x1:input的列
    if(y1<M && x1<N){
       S[threadIdx.y][threadIdx.x] = input[y1 * N + x1];  // 合并读取
       //warp内相邻线程访问input中的连续地址，故而GPU可以把这些线程的
       //读取操作合并，从而加快读取速度
    }
    __syncthreads();

    const int x2=by+threadIdx.x;
    const int y2=bx+threadIdx.y;

    //x2:output的列
    if(y2<N && x2<M)
        output[y2*M+x2]=S[thradIdx.x][threadIdx.y];//合并写入
    //这里如果没有padding,则同一个warp中现成的threadIdx.y相同
    //线程们会被映射给同一个bank，从而发生bank conflict
    //类似的,这里的相邻线程(同一个warp内的相邻线程的threadIdx.x是
    //连续的),把写入output的操作合并,因为2写入的地址也是连续的.

    //output[y2][x2] 这里只看x2,x2=by+threadIdx.x
    //而后面的S[threadIdx.x]...我们只看threadIdx.x
    //S在写入时，前面是对于的threadIdx.y=by+threadIdx.y
    //另一个进程的threadIdx.y对应当前线程后面的threadIdx.x
    //故而这里output的x2=by+threadIdx.x=input中的y1
    //从而实现置换
    }
}

// 使用共享内存中转，合并读取+写入，使用swizzling解决bank conflict
template<BLOCKSIZE>
__global__ void transpose_v4(float* input,float* output,int M,int N){
    __shared__ float S[BLOCKSIZE][BLOCKSIZE];
    int bx=blockDim.x*blockIdx.x;
    int by=blockDim.y*blockIdx.y;

    int x1=bx+threadIdx.x;
    int y1=by+threadIdx.y;

    if(y1<M && x1<N){
        S[threadIdx.y][threadIdx.x^threadIdx.y]=input[y1*N+x1];
    }
    __syncthreads();

    int x2=by+threadIdx.x;
    int y2=bx+threadIdx.y;
    if(y2<N &&x2<M){
        output[y2*M+x2]=S[threadIdx.x][threadIdx.x^threadIdx.y];//合并写入
        //0至31依次与同一个数做异或，还是会被映射到0至31
        //但是可以避免bank conflict
        //因为连续线程对应的S中的列不再一样，同时由于映射
        //是一对一的，因此还可以找到正确的数据
        //a ^ b= b ^ a (异或具有交换律)
    }
}

int main() {
    // 输入是M行N列，转置后是N行M列
    size_t M = 12800;
    size_t N = 1280;
    constexpr size_t BLOCK_SIZE = 32;
    const int repeat_times = 10;

    // --------------------host 端计算一遍转置, 输出的结果用于后续验证---------------------- //
    float *h_matrix = (float *)malloc(sizeof(float) * M * N);
    float *h_matrix_tr_ref = (float *)malloc(sizeof(float) * N * M);
    randomize_matrix(h_matrix, M * N);
    host_transpose(h_matrix, M, N, h_matrix_tr_ref);
    // printf("init_matrix:\n");
    // print_matrix(h_matrix, M, N);
    // printf("host_transpose:\n");
    // print_matrix(h_matrix_tr_ref, N, M);

    float *d_matrix;
    cudaMalloc((void **) &d_matrix, sizeof(float) * M * N);
    cudaMemcpy(d_matrix, h_matrix, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    free(h_matrix);

    // --------------------------------call transpose_v0--------------------------------- //
    float *d_output0;
    cudaMalloc((void **) &d_output0, sizeof(float) * N * M);                              // device输出内存
    float *h_output0 = (float *)malloc(sizeof(float) * N * M);                            // host内存, 用于保存device输出的结果

    dim3 block_size0(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size0(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));                            // 根据input的形状(M行N列)进行切块
    float total_time0 = TIME_RECORD(repeat_times, ([&]{device_transpose_v0<<<grid_size0, block_size0>>>(d_matrix, d_output0, M, N);}));
    cudaMemcpy(h_output0, d_output0, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output0, h_matrix_tr_ref, M * N);                                     // 检查正确性
    printf("[device_transpose_v0] Average time: (%f) ms\n", total_time0 / repeat_times);  // 输出平均耗时

    cudaFree(d_output0);
    free(h_output0);

    // --------------------------------call transpose_v1--------------------------------- //
    float *d_output1;
    cudaMalloc((void **) &d_output1, sizeof(float) * N * M);                              // device输出内存
    float *h_output1 = (float *)malloc(sizeof(float) * N * M);                            // host内存, 用于保存device输出的结果

    dim3 block_size1(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size1(CEIL(M, BLOCK_SIZE), CEIL(N, BLOCK_SIZE));                            // 根据output的形状(N行M列)进行切块
    float total_time1 = TIME_RECORD(repeat_times, ([&]{device_transpose_v1<<<grid_size1, block_size1>>>(d_matrix, d_output1, M, N);}));
    cudaMemcpy(h_output1, d_output1, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output1, h_matrix_tr_ref, M * N);                                     // 检查正确性
    printf("[device_transpose_v1] Average time: (%f) ms\n", total_time1 / repeat_times);  // 输出平均耗时

    cudaFree(d_output1);
    free(h_output1);

    // --------------------------------call transpose_v2--------------------------------- //
    float *d_output2;
    cudaMalloc((void **) &d_output2, sizeof(float) * N * M);                              // device输出内存
    float *h_output2 = (float *)malloc(sizeof(float) * N * M);                            // host内存, 用于保存device输出的结果

    dim3 block_size2(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size2(CEIL(M, BLOCK_SIZE), CEIL(N, BLOCK_SIZE));                            // 根据output的形状(N行M列)进行切块
    float total_time2 = TIME_RECORD(repeat_times, ([&]{device_transpose_v2<<<grid_size2, block_size2>>>(d_matrix, d_output2, M, N);}));
    cudaMemcpy(h_output2, d_output2, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output2, h_matrix_tr_ref, M * N);                                     // 检查正确性
    printf("[device_transpose_v2] Average time: (%f) ms\n", total_time2 / repeat_times);  // 输出平均耗时

    cudaFree(d_output2);
    free(h_output2);

    // --------------------------------call transpose_v3--------------------------------- //
    float *d_output3;
    cudaMalloc((void **) &d_output3, sizeof(float) * N * M);                              // device输出内存
    float *h_output3 = (float *)malloc(sizeof(float) * N * M);                            // host内存, 用于保存device输出的结果

    dim3 block_size3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size3(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));                            // 根据input的形状(M行N列)进行切块
    float total_time3 = TIME_RECORD(repeat_times, ([&]{device_transpose_v3<BLOCK_SIZE><<<grid_size3, block_size3>>>(d_matrix, d_output3, M, N);}));
    cudaMemcpy(h_output3, d_output3, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output3, h_matrix_tr_ref, M * N);                                     // 检查正确性
    printf("[device_transpose_v3] Average time: (%f) ms\n", total_time3 / repeat_times);  // 输出平均耗时

    cudaFree(d_output3);
    free(h_output3);

    // --------------------------------call transpose_v4--------------------------------- //
    float *d_output4;
    cudaMalloc((void **) &d_output4, sizeof(float) * N * M);                              // device输出内存
    float *h_output4 = (float *)malloc(sizeof(float) * N * M);                            // host内存, 用于保存device输出的结果

    dim3 block_size4(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size4(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));                            // 根据input的形状(M行N列)进行切块
    float total_time4 = TIME_RECORD(repeat_times, ([&]{device_transpose_v4<BLOCK_SIZE><<<grid_size4, block_size4>>>(d_matrix, d_output4, M, N);}));
    cudaMemcpy(h_output4, d_output4, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output4, h_matrix_tr_ref, M * N);
    printf("[device_transpose_v4] Average time: (%f) ms\n", total_time4 / repeat_times);

    cudaFree(d_output4);
    free(h_output4);

    // ---------------------------------------------------------------------------------- //
    free(h_matrix_tr_ref);

    return 0;
}