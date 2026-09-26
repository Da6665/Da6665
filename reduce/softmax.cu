//归约求softmax

//当多个线程执行时，返回max(*address,val_1,val_2,...)
__device__ static float atomicMax(float* address,float val){
    int* address_as_i=(int*) address;
    int old=*address_as_i;
    int assumed;
    do{
        assumed=old;
        old=atomicCAS(address_as_i,assumed,__float_as_int(fmax(val,__int_as_float(address_as_i)));
        //相当于
        // if(*address_as_i==assumed){
        //     *address_as_i=____float_as_int(fmax(val,__int_as_flaot(*address);}
        //old=*address_as_i;
        //因为初始的*address_as_i 可能已经被其它线程修改
    }while(old!=assumed);
    return __int_as_float(old);
}

//规约求最值
__global__ void max_kernel(float* input,float* max_val,const int N){
    __shared__ float s_men[32];
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    int warpid=threadIdx.x/warpSize;
    int laneid=threadIdx.x% warpSize;

    float val=(int<N)? input[idx]:0.0f;

    #pragma unroll
    for(int offset=warpSize>>1;offset>0;offset>>=1){
        val=fmax(val,__shlf_down_sync(oxffffffff,val,offset));
    }
    if(laneid==0)   s_men[warpid]=val;
    __syncthreads();

    if(warpid==0){
        int warpNum=blockDim.x/warpSize;
        val=(laneid<warpNum)? s_men[laneid]:0.0f;

        #pragma unroll
        for(int offset=warpSize>>1;offset>0;offset>>=1){
            val=fmax(val,__shlf_down_sync(oxffffffff,val,offset));
        }

        if(laneid==0)  atomicMax(max_val,val);
    }
}

//归约求最和
__global__ void sum_kernel(float* input,float* sum,float* max_val,int N){
    __shared__ float s_men[32];
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    int warpid=threadIdx.x/warpSize;
    int laneid=threadIdx.x% warpSize;

    float val=(idx<N)? expf(input[idx]-*max_val):0.0f;

    #pragma unroll
    for(int offset>>1;offset>0;offset>>=1){
        val+=__shlf_down_sync(0xffffffff,val,offset);
    }

    if(laneid==0)  s_men[warpid]=val;
    __syncthreads();

    if(warpid==0){
        int warpNum=blockDim.x/warpSize;
        val=(laneid<warpNum)? s_men[laneid]:0.0f;
        for(int offset=warpSize>>1;offset>0;offset>>=1){
            val+=__shlf_down_sync(0xffffffff,val,offset);
        }
        if(laneid==0)  atomicAdd(sum,val);
    }
}

//softmax函数
__global__ void softmax_kernel(float* input,float* output,float* sum,float* max_val,int N){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<N) output[idx]=expf(inut[idx]-*max_val)/(*sum);
}


//softmax cpu实现版本
void softmax(float* input,float* output,int N){
    float* M=*(std::max_element(input,input+N));
    float sum=0.0;
    for(int i=0;i<N;i++){
        output[idx]=expf(input[idx]-M);
        sum+=output[idx];
    }
    for(int i=0;i<N;i++){
        output[idx]/=sum;
    }
}

//调用
int main(){
    int blockSize=256;
    int N=1024;
    int gridSize=CEIL(N,blockSize);
    float* max_val,sum;
    max_kernel<<<gridSize,blockSize>>>(input,max_val,N);
    sum_kernel<<<gridSize,blockSize>>>(input,sum,max_val,N);
    softmax_kernel<<<gridSize,blockSize>>>(input,output,sum,max_val,N);
}

