#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cxtimers.h"

// 使用泰勒級數展開計算正弦函數

__host__ __device__ inline float sinsum(float x, int terms){

    float term = x;
    float sum  = term;
    float x2 = x*x;

    // 循環計算每一項，並加到總和中
    for(int n = 1; n < terms ; n++){
        term *= -x2 / (float)(2*n*(2*n+1));
        sum += term;
    }
    return sum;
}

// GPU上執行的核心函數，計算一系列步驟中的正弦和
__global__ void gpu_sin(float *sums, int steps, int terms, float step_size) {
    // 計算當前執行緒應處理的步驟

    int step = blockIdx.x * blockDim.x + threadIdx.x;

    // 確保步驟在範圍內
    if (step < steps) {
        float x = step_size * step;
        sums[step] = sinsum(x, terms);
    }
}

// 進行reduction，將陣列上的值總和為一個值
__global__ void reductionKernel(float *data, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    // 加載數據到共享記憶體
    sdata[tid] = (index < n) ? data[index] : 0;
    __syncthreads();

    // Perform reduction in shared memory
    // 在共享記憶體加總
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    // 把每個block的結果寫入global memory
    if (tid == 0) {
        data[blockIdx.x] = sdata[0];
    }
}


double gpuReduction(float *d_sums, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    while (blocks > 1) {
        int sharedSize = threads * sizeof(float);
        reductionKernel<<<blocks, threads, sharedSize>>>(d_sums, n);
        cudaDeviceSynchronize();

        n = blocks;
        blocks = (n + threads - 1) / threads;
    }

    float result;
    cudaMemcpy(&result, d_sums, sizeof(float), cudaMemcpyDeviceToHost);
    return (double)result;
}

int main(int argc, char *argv[]) {

    int steps = (argc > 1) ? atoi(argv[1]) : 10000000;
    int terms = (argc > 2) ? atoi(argv[2]) : 10000;

    double pi = 3.1415952535897323;
    double step_size = pi / (steps - 1);

    float *d_sums;
    float *h_sums = (float *)malloc(sizeof(float) * steps);
    
    cx::timer tim;

    // Allocate memory on the GPU
    cudaMalloc(&d_sums, sizeof(float) * steps);
    
    // Execute the kernel
    int threads = 256;  // Assuming 256 threads per block
    int blocks = (steps + threads - 1) / threads;
    
    gpu_sin<<<blocks, threads>>>(d_sums, steps, terms, step_size);
    
    // Transfer the result back to host
    cudaMemcpy(h_sums, d_sums, sizeof(float) * steps, cudaMemcpyDeviceToHost);

    double gpu_sum = gpuReduction(d_sums, steps);
    
    double gpu_time = tim.lap_ms();

    // Correction and final calculation
    gpu_sum -= 0.5 * (sinsum(0.0f, terms) + sinsum(pi, terms));
    gpu_sum *= step_size;
    
    printf("gpu sum %.10f steps %d terms %d time %.3f ms\n",gpu_sum,steps,terms,gpu_time);

    // FREEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    cudaFree(d_sums);
    free(h_sums);

    return 0;
}
