#include <iostream>
#include <cuda_runtime.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))
#define FLOAT4(value)  (reinterpret_cast<float4*>(&value)[0])

__global__ void relu(float* x,float* y,int N){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<N){
        y[idx]=fmaxf(x[idx],0.0f);
    }
}

__global__ void relu_float4(float* x,float* y,int N){
    int idx=(blockDim.x*blockIdx.x+threadIdx.x)*4;
    if(idx<N){
        float4 tempx=FLOAT4(x[idx]);
        flaot4 tempy;
        tempy.x=fmaxf(tempx.x,0.0f);
        tempy.y=fmaxf(tempx.y,0.0f);
        tempy.z=fmaxf(tempx.z,0.0f);
        tempy.w=fmaxf(tempx.w,0.0f);
        y[idx]=FLOAT4(tempy);
    }
}