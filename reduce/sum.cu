//规约求和操作
#include <stdio.h>
#include <stdlib.h>
#include "utils.cuh"

void host_reduce(float* x,int N,float* sum){
    *sum=0.0f;
    for(int i=0;i<N;i++){
        *sum+=x[i];
    }
}
//global函数每一个线程都会执行一次
//而每一个线程都有唯一的线程ID，线程ID由blockIdx和threadIdx组成

//reduce_v0使用共享内存来操作，即原地操作，原来的向量(数组d_x)中数值会改变
__global__ void device_redce_v0(float* d_x,float* d_y){
    const int tid=threadIdx.x;//当前线程在block块中的线程ID
    float* x=&d_x[blockIdx.x*blockDim.x];//当前block块所处理元素的首地址

    for(int offset=blockDim.x>>1;offset>0;offset>>=1){
        if(tid<offset){
            x[tid]+=x[tid+offset];
        }
        __syncthreads();//同步线程，保证所有线程都执行完毕
        //因为下一轮累计需要用到上一轮的结果，所以必须等上一轮的线程
        //全部执行完毕，才能进行下一轮操作
    }
    if(tid==0){
        d_y[blockIdx.x]=x[0];//将每个block块的结果写入全局内存
    }
}
template<const int BLOCK_SIZE>
 void call_reduce_v0(float* d_x,float* d_y,float* h_y,const int N,float* sum){
    const int GRID_SIZE=CEIL(N,BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(GRID_SIZE);
    device_reduce_v0<<<grid_size,block_size>>>(d_x,d_y);
    cudaMencpy(h_y,d_y,GRID_SIZE*sizeof(float),cudaMemcpyDeviceToHost);
    //主机端再次规约一遍
    *sum=0.0f;
    for(int i=0;i<GRID_SIZE;i++){
        *sum+=h_y[i];
    }
}

template<const int BLOCK_SIZE>
__global__ void device_reduce_v1(float* d_x,float* d_y,const int N){
    const int tid=threadIdx.x;
    const int bid=blockIdx.x;
    const int n=bid*blockDim.x+tid;
    __shared__float s_y[BLOCK_SIZE];
    //在每个block块中开辟一个大小为BLOCK_SIZE的共享内存数组s_y
    //所有的线程都共享这个s_y数组
    s_y[tid]=(n<N)?d_x[n]:0.0f;
    __syncthreads();
    //syncthreads()：线程同步函数，线程必须等待同一个block中所有线程都将
    //__syncthreads()函数前的代码执行完毕，才能继续执行后续代码
    //线程同步，保证每个block中所有线程的数据都copy到 s_y中

    for(int offset=blockDim.x>>1;offset>0;offset>>=1){
        if(tid<offset){
            s_y[tid]+=s_y[tid+offset];
        }
        __syncthreads();
    }
    if(tid==0){
        //只有block中线程ID为0的线程才会执行下面的代码
        d_y[bid]=s_y[0];
    }
}


template<const int BLOCK_SIZE>
__global__ void device_reduce_v2(float* d_x,float* d_y,float* h_y,const int N,float* sum){
    const int GRID_SIZE=CEIL(N,BLOCK_SIZE);
    //运行时，BLOCK_SIZE是一个常量，编译器会将其替换为具体的数值
    //GRID_SIZE也是一个常量
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(GRID_SIZE);
    device_reduce_v1<<<grid_size,block_size>>>(d_x,d_y,N);
    cudaMemcpy(h_y,d_y,GRID_SIZE*sizeof(float),cudaMemcpyDeviceToHost);
    *sum=0.0f;
    for(int i=0;i<GRID_SIZE;i++){
        *sum+=h_y[i];
    }
}

// reduce_v3：改进，引入原子函数，不需要再到CPU进行归约了
__global__ void device_reduce_v3(float* d_x, float* d_y, const int N) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ float s_y[];  // 动态共享内存
    s_y[tid] = (n < N) ? d_x[n] : 0.0;  // 搬运global mem 到 shared mem
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(d_y, s_y[0]);  // 原子函数，将取出*d_y，与s_y[0]求和后，再根据地址d_y写回去
        // *d_y += s_y[0];  // 错误，因为d_y如果被多个线程同时读取，再写入时结果就会发生错误

        //因为加上了tid==0，所以每个block只有一个线程会执行这条语句，避免了多个线程同时访问d_y的情况
        //原子函数可以让每一个时刻只有一个线程访问d_y，避免了多个线程同时访问d_y的情况，从而保证了结果的正确性
    }
}

template <const int BLOCK_SIZE>
void call_reduce_v3(float* d_x, float* d_y, float* h_y, const int N) {
    const int GRID_SIZE = CEIL(N, BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(GRID_SIZE);
    *h_y = 0.0;  // host端d_y清零
    cudaMemcpy(d_y, h_y, sizeof(float), cudaMemcpyHostToDevice);  // 拷贝给d_y
    device_reduce_v3<<<grid_size, block_size, sizeof(float) * BLOCK_SIZE>>>(d_x, d_y, N);  // 使用（动态）共享内存
    cudaMemcpy(h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);  // 拷贝回h_y
    cudaDeviceSynchronize();
}

__global__ void device_reduce_v4(float* d_x,float* d_y,const int N){
    __shared__ float s_y[32];

    int idx=blockDim.x*blockIdx.x+threadx.x;
    int warpId=threadIdx.x/warpSize;
    int laneId=threadIdx.x% warpSize;
    float value=(idx<N)?d_x[idx]:0.0f;

    #pragma unroll
    for(int offset=warpSize>>1;offset>0;offset>>=1){
        value+=__shlf_down_sync(0xffffffff,value,offset);
        //__shlf_down_sync()会等一个warp中的线程把前面的操作都执行完
        //然后等待所有的线程把数据value都准备好
        //然后再传递给对应的线程，这样每个线程都能正确接收自己所需要的线程
        //oxffffffff:表示所有的线程都参加
        //后面的参数:把当前线程后面第offset个线程的value值传递给当前线程
    }
    if(laneId==0)  s_y[warpId]=value;
    __syncthreads();

    if(warpId==0){
        int warpNum=blockDim.x/warpSize;
        value=(laneId<warpNum)? s_y[laneId]:0.0f;

        #pragma unroll
        for(int offset=warpSize>>1;offset>0;offset>>=1){
            value+=__shlf_down_sync(0xffffffff,value,offset);
        }
        if(laneId==0)  atomicAdd(d_y,value);
        //d_y表示变量的地址
        //*d_y表示地址中的值，即d_y
    }
}

template <const int BLOCK_SIZE>
void call_reduce_v4(float* d_x, float* d_y, float* h_y, const int N) {
    const int GRID_SIZE = CEIL(N, BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(GRID_SIZE);
    *h_y = 0.0;  // host端d_y清零
    cudaMemcpy(d_y, h_y, sizeof(float), cudaMemcpyHostToDevice);  // 拷贝给d_y
    device_reduce_v4<<<grid_size, block_size>>>(d_x, d_y, N);  // 使用（动态）共享内存
    cudaMemcpy(h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);  // 拷贝回h_y
    cudaDeviceSynchronize();
}

__global__ void device_reduce_v5(float* d_x,float* d_y,const int N){
    __shared__ float s_y[32];
    int idx=(blockDim.x*blockIdx.x+threadIdx.x)*4;
    //每个线程处理d_x中的4个数据
    int wrapid=blockDim.x/warpSize;
    int laneid=blockDim.x%warpSize;
    float val=0.0f;
    if(idx<N){
        float4 temp=FLOAT4(d_x[idx]);
        val+=temp.x;
        val+=temp.y;
        val+=temp.z;
        val+=temp.w;
    }

    for(int offset=warpSize>>1;offset>0;offset>>=1){

        val+=__shlf_down_sync(oxffffffff,val,offset);
    }
    if(laneid==0) s_y[wrapid]=val;
    __syncthreads()

    if(wrapid==0){
        int wrapNum=blockDim.x/wrapSize;
        val=(laneid<warpNum)? s_y[laneid]:0.0f;

        for(int offset=wrapSize>>1;offset>0;offset>>=1){

            val+=__shlf_down_sync(oxffffffff,val,offset);
        }
        if(laneid==0)  atomicAdd(d_y,val);
    }
}

template <const int BLOCK_SIZE>
void call_reduce_v5(float* d_x, float* d_y, float* h_y, const int N) {
    const int GRID_SIZE = CEIL(CEIL(N, BLOCK_SIZE), 4);  // 这里要除以4
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(GRID_SIZE);
    *h_y = 0.0;  // host端d_y清零
    cudaMemcpy(d_y, h_y, sizeof(float), cudaMemcpyHostToDevice);  // 拷贝给d_y
    device_reduce_v5<<<grid_size, block_size>>>(d_x, d_y, N);  // 使用（动态）共享内存
    cudaMemcpy(h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);  // 拷贝回h_y
    cudaDeviceSynchronize();
}



int main() {
    size_t N = 100000000;
    constexpr size_t BLOCK_SIZE = 128;
    const int repeat_times = 10;

    // 1. host
    float *h_nums = (float *)malloc(sizeof(float) * N);
    float *sum = (float *)malloc(sizeof(float));
    randomize_matrix(h_nums, N);
    
    float total_time_h = TIME_RECORD(repeat_times, ([&]{host_reduce(h_nums, N, sum);}));
    // printf("init_matrix:\n");
    // print_matrix(h_nums, 1, N);
    printf("[reduce_host]: sum = %f, total_time_h = %f ms\n", *sum, total_time_h / repeat_times);

    // 2. device
    float *d_nums, *d_rd_nums;
    cudaMalloc((void **) &d_nums, sizeof(float) * N);
    cudaMalloc((void **) &d_rd_nums, sizeof(float) * CEIL(N, BLOCK_SIZE));
    float *h_rd_nums = (float *)malloc(sizeof(float) * CEIL(N, BLOCK_SIZE));
    
    // 2.1 call reduce_v0, 全局内存，因为reduce会把归约结果累加到d_nums（global memory）上，所以重复执行reduce_v0，得到的sum会越来越大
    cudaMemcpy(d_nums, h_nums, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_0 = TIME_RECORD(repeat_times, ([&]{call_reduce_v0<BLOCK_SIZE>(d_nums, d_rd_nums, h_rd_nums, N, sum);}));
    printf("[reduce_v0]: sum = %f, total_time_0 = %f ms\n", *sum, total_time_0 / repeat_times);

    // 2.2 call reduce_v1，使用静态共享内存，重复执行，sum不受影响
    cudaMemcpy(d_nums, h_nums, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_1 = TIME_RECORD(repeat_times, ([&]{call_reduce_v1<BLOCK_SIZE>(d_nums, d_rd_nums, h_rd_nums, N, sum);}));
    printf("[reduce_v1]: sum = %f, total_time_1 = %f ms\n", *sum, total_time_1 / repeat_times);    

    // 2.3 call reduce_v2，在v1基础上改成动态共享内存，性能维持不变
    cudaMemcpy(d_nums, h_nums, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_2 = TIME_RECORD(repeat_times, ([&]{call_reduce_v2<BLOCK_SIZE>(d_nums, d_rd_nums, h_rd_nums, N, sum);}));
    printf("[reduce_v2]: sum = %f, total_time_2 = %f ms\n", *sum, total_time_2 / repeat_times);

    // 2.4 call reduce_v3，在v2基础上引入原子函数，不需要再到CPU进行归约了
    float *d_sum;
    cudaMalloc((void **) &d_sum, sizeof(float));
    cudaMemcpy(d_nums, h_nums, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_3 = TIME_RECORD(repeat_times, ([&]{call_reduce_v3<BLOCK_SIZE>(d_nums, d_sum, sum, N);}));
    printf("[reduce_v3]: sum = %f, total_time_3 = %f ms\n", *sum, total_time_3 / repeat_times);    

    // 2.5 call reduce_v4，使用warp shuffle
    cudaMemcpy(d_nums, h_nums, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_4 = TIME_RECORD(repeat_times, ([&]{call_reduce_v4<BLOCK_SIZE>(d_nums, d_sum, sum, N);}));
    printf("[reduce_v4]: sum = %f, total_time_4 = %f ms\n", *sum, total_time_4 / repeat_times);    

    // 2.6 call reduce_v5，使用warp shuffle + float4
    cudaMemcpy(d_nums, h_nums, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_5 = TIME_RECORD(repeat_times, ([&]{call_reduce_v5<BLOCK_SIZE>(d_nums, d_sum, sum, N);}));
    printf("[reduce_v5]: sum = %f, total_time_5 = %f ms\n", *sum, total_time_5 / repeat_times);    

    // free memory
    free(h_nums);
    free(sum);
    free(h_rd_nums);
    cudaFree(d_nums);
    cudaFree(d_rd_nums);
    cudaFree(d_sum);
    return 0;
}




