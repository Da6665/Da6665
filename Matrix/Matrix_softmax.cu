#include<cuda_runtime.h>
#include <flaot.h>
#include <math.h>




__device__ __forceinline__
float warpReduceMax(float val){
    #pragma unroll
    for(int offset=warpSize>>1;offset>0;offset>>=1){
        val=fmax(val,__shfl_down_sync(0xffffffff,val,offset));
    }
    return val;
}

float warpReduceSum(float val){
    #pragma unroll
    for(int offset=warpSize>>1;offset>0;offset>>=1){
        val+=__shfl_down_sync(0xffffffff,val,offset);
    }
    return val;
}

//求矩阵某一行的softmax
//一个block负责一行
//矩阵:M行N列
__global__ void softmax_row(float* input,float* output,int M,int N){
    __shared__ float S[32];
    __shared__ float block_max;
    __shared__ float blcok_sum;

    int row=blockIdx.x;
    if(row>=M) return;

    int laneId=threadIdx.x % warpSize;
    int warpId=threadIdx.x /warpSize;
    int warpNum=blockDim.x /warpSize;
    int tid=threadIdx.x;

    float max_val=-FLT_MAX;
    for(int col=tid;col<N;col+=blockDim.x){
        max_val=fmax(val,input[row*N+col];
    }
    
    //每个warp内部Reduce
    max_val=warpReduceMax(max_val);
    if(laneId==0) S[warpId]=max_val;
    __syncthreads();

    //warp0内对所有的warp结果再进行规约
    if(warp==0){
        max_val=(laneId<warpNum)? S[laneId] : -FLT_MAX;
        max_val=warpReduceMax(max_val);

        if(laneId==0) block_max=max_val;
    }

    __syncthreads();
    max_val=block_max;

    //计算exp的局部sum
    float sum=0.0f;

    for(col=tid;col<N;col+=blockDim.x){
        sum+=expf(input[row*N+col]-max_val);
    }

    //一个warp内规约
    sum=warpRudecesum(sum);
    if(laneId==0) S[warpId]=sum;
    __syncthreads();

    if(warpId==0){
        sum=(laneId<warpNum)? S[laneId]:0.0f;
        sum=warpReduceSum(sum);

        if(laneId==0) block_sum=sum;
    }

    __syncthreads();
    sum=block_sum;

    //输出softmax
    for(int col=tid;col<N;col+=blockDim.x){
        output[row*N+col]=expf(input[row*N+col]-max_val)/sum;
    }
}

//按列进行softmax，如果按照上面类似的操作直接访问input和output
//由于相邻线程访问的位置不连续，无法合并访问
//用共享内存做优化
//一个block处理32列

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define TILE 32


// warp 内求最大值
__device__ __forceinline__
float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(
            val,
            __shfl_down_sync(0xffffffff, val, offset)
        );
    }
    return val;
}


// warp 内求和
__device__ __forceinline__
float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(
            0xffffffff,
            val,
            offset
        );
    }
    return val;
}


// GPU：对矩阵每一列做 softmax
//
// 一个 block 处理连续 32 列
// blockDim = (32, 32)
//
// threadIdx.x：laneId，也表示 tile 中的列
// threadIdx.y：warpId
//
__global__ void softmax_col_shared(
    const float* input,
    float* output,
    int M,
    int N
) {

    // =====================================================
    // Shared Memory
    //
    // 多加 1 列 padding：
    // [32][33] 而不是 [32][32]
    //
    // 避免后面按列访问时发生 bank conflict
    // =====================================================

    __shared__ float tile[TILE][TILE + 1];

    __shared__ float s_max[TILE];
    __shared__ float s_sum[TILE];


    int laneId = threadIdx.x;   // 0 ~ 31
    int warpId = threadIdx.y;   // 0 ~ 31

    // 当前 block 负责的 32 列的起点
    int col_base = blockIdx.x * TILE;

    // 当前 warp 负责哪一列
    int col = col_base + warpId;



    // =====================================================
    // 第一部分：求每一列最大值
    //
    // 一个 warp 负责一列
    // =====================================================

    float max_val = -FLT_MAX;


    // 每次处理 32 行
    for (int row_base = 0;
         row_base < M;
         row_base += TILE) {


        // -------------------------------------------------
        // 1. Global Memory -> Shared Memory
        //
        // 每个 warp 按行连续读取 32 个数据
        // 所以 global memory 可以合并访问
        // -------------------------------------------------

        int load_row = row_base + threadIdx.y;
        int load_col = col_base + threadIdx.x;

        if (load_row < M && load_col < N) {

            tile[threadIdx.y][threadIdx.x]
                = input[load_row * N + load_col];

        } else {

            tile[threadIdx.y][threadIdx.x]
                = -FLT_MAX;
        }


        __syncthreads();


        // -------------------------------------------------
        // 2. 每个 warp 转过来读取一列
        //
        // warpId 决定列
        // laneId 决定行
        //
        // warp0：
        // tile[0][0]
        // tile[1][0]
        // ...
        // tile[31][0]
        //
        // warp1：
        // tile[0][1]
        // tile[1][1]
        // ...
        // -------------------------------------------------

        if (col < N) {

            float val =
                tile[laneId][warpId];

            val = warpReduceMax(val);


            // warpReduceMax 最后的结果在 lane0
            if (laneId == 0) {
                max_val =
                    fmaxf(max_val, val);
            }
        }


        // 防止下一轮覆盖 tile
        __syncthreads();
    }


    // lane0 保存这一列最终最大值
    if (laneId == 0 && col < N) {
        s_max[warpId] = max_val;
    }


    __syncthreads();



    // =====================================================
    // 第二部分：计算
    //
    // sum(exp(x - max))
    // =====================================================

    float sum = 0.0f;


    for (int row_base = 0;
         row_base < M;
         row_base += TILE) {


        // -------------------------------------------------
        // 1. 再次合并读取一个 32×32 tile
        // -------------------------------------------------

        int load_row = row_base + threadIdx.y;
        int load_col = col_base + threadIdx.x;

        if (load_row < M && load_col < N) {

            tile[threadIdx.y][threadIdx.x]
                = input[load_row * N + load_col];

        }

        __syncthreads();


        // -------------------------------------------------
        // 2. 一个 warp 计算一列
        // -------------------------------------------------

        int row = row_base + laneId;

        float val = 0.0f;

        if (row < M && col < N) {

            val = expf(
                tile[laneId][warpId]
                - s_max[warpId]
            );
        }


        val = warpReduceSum(val);


        // lane0 累加当前这 32 行的结果
        if (laneId == 0 && col < N) {
            sum += val;
        }


        __syncthreads();
    }


    // 保存整列最终 sum
    if (laneId == 0 && col < N) {
        s_sum[warpId] = sum;
    }


    __syncthreads();



    // =====================================================
    // 第三部分：计算最终 softmax
    //
    // exp(x - max) / sum
    // =====================================================

    for (int row_base = 0;
         row_base < M;
         row_base += TILE) {


        // 这里重新按照行连续写出
        //
        // 一个 warp 写连续 32 个元素
        // 所以 global memory 写也是合并的

        int row = row_base + threadIdx.y;
        int output_col = col_base + threadIdx.x;


        if (row < M && output_col < N) {

            float x =
                input[row * N + output_col];

            output[row * N + output_col] =
                expf(
                    x - s_max[threadIdx.x]
                )
                / s_sum[threadIdx.x];
        }
    }
}