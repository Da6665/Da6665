#include <iostream>
#include <cuda_runtime.h>

#define CEIL (a,b)  (((a)+(b)-1)/(b))
#define FLOAT4(value)  (reinterpret_cast<float4*>(&value)[0])

__global__void sigmoid(float* x,float* y,int N){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<N){
        y[idx]=1.0f/(1.0f+expf(-x[idx]));
    }
}

__global__void sigmoid_float4(float* x,float* y,int N){
    int idx=(blockDim.x*blockIdx.x+threadIdx.x)*4;
    if(idx<N){
        float4 tempx=FLOAT4(x[idx]);
        float4 tempy;
        tempy.x=1.0f/(1.0f+expf(-tempx.x));
        tempy.y=1.0f/(1.0f+expf(-tempx.y));
        tempy.z=1.0f/(1.0f+expf(-tempx.z));
        tempy.w=1.0f/(1.0f+expf(-tempx.w));
        y[idx]=FLOAT4(tempy);
    }
}