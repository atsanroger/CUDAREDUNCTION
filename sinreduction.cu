#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cxtimers.h"

typedef struct {
    int maxThreadsPerBlock;
    size_t sharedMemPerBlock;
} GpuProperties;

void getGpuProperties(GpuProperties *gpuProps) {

    int device;
    cudaDeviceProp prop;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    gpuProps->maxThreadsPerBlock = prop.maxThreadsPerBlock;
    gpuProps->sharedMemPerBlock  = prop.sharedMemPerBlock;
}

__host__ __device__ inline double sinsum(double x, int terms){

    double term = x;
    double sum  = term;
    double x2 = x*x;

    for(int n = 1; n < terms ; n++){
        term *= -x2 / (double)(2*n*(2*n+1));
        sum += term;
    }
    return sum;
}

__global__ void gpu_sin(double *sums, int steps, int terms, double step_size) {
    extern __shared__ double sharedSums[];

    int step = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (step < steps) {
        double x = step_size * step;
        sharedSums[tid] = sinsum(x, terms);
    } else {
        sharedSums[tid] = 0.0;
    }
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedSums[tid] += sharedSums[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        sums[blockIdx.x] = sharedSums[0];
    }
}

__global__ void reductionKernel(double *data, int n) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (index < n) ? data[index] : 0;
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        data[blockIdx.x] = sdata[0];
    }
}


double gpuReduction(double *d_sums, int n, GpuProperties gpuProps) {

    int threads = min(256, gpuProps.maxThreadsPerBlock);
    int blocks = (n + threads - 1) / threads;

    while (blocks > 1) {
        int sharedSize = threads * sizeof(double);
        reductionKernel<<<blocks, threads, sharedSize>>>(d_sums, n);
        cudaDeviceSynchronize();

        n = blocks;
        blocks = (n + threads - 1) / threads;
    }

    double result;
    cudaMemcpy(&result, d_sums, sizeof(double), cudaMemcpyDeviceToHost);
    return (double)result;
}

int main(int argc, char *argv[]) {

    GpuProperties gpuProps;
    getGpuProperties(&gpuProps);

    int steps = (argc > 1) ? atoi(argv[1]) : 65600;
    int terms = (argc > 2) ? atoi(argv[2]) : 10000;

    constexpr double pi = 3.1415952535897323;
    double step_size = pi / (steps - 1);

    double *d_sums;

    cx::timer tim;

    // Allocate memory on the GPU
    cudaMalloc(&d_sums, sizeof(double) * steps);
    
    // Execute the kernel
    int threadsForSin  = min(256, gpuProps.maxThreadsPerBlock);  
    int blocksForSin  = (steps + threadsForSin - 1) / threadsForSin;
    
    gpu_sin<<<blocksForSin, threadsForSin, sizeof(double) * threadsForSin>>>(d_sums, steps, terms, step_size);
    
    double gpu_sum = gpuReduction(d_sums, steps, gpuProps);
    double gpu_time = tim.lap_ms();

    // Correction and final calculation
    gpu_sum -= 0.5 * (sinsum(0.0f, terms) + sinsum(pi, terms));
    gpu_sum *= step_size;
    
    printf("gpu sum %.10lf steps %d terms %d time %.3f ms\n", gpu_sum, steps, terms,gpu_time);

    // FREEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
    cudaFree(d_sums);

    return 0;
}
