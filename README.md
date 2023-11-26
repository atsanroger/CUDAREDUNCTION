# CUDAReduction

## Introduction
CUDAReduction is a high-performance computing project focused on calculating the integral of the sine function using CUDA for GPU acceleration. 
The project compares the results and performance of GPU-based calculations with a CPU version that utilizes both OpenMP and MPI for parallel processing.
Currently the GPU code gives wrong result.

## Requirements
- MPI
- OpenMP
- CUDA
- A CUDA-compatible GPU

## Compilation and Execution

### CPU Version
To compile the CPU version of the program, use the following command:
```
mpic++ -fopenmp reductionCPU.cpp -o reductionCPU.o -lm
```
To execute the compiled CPU program, use:

```
OMP_NUM_THREADS=4 mpirun -np 8 ./reductionCPU.o
```

Modify the number of thread and prcoessor according to your hardware/

### GPU Version
For the GPU version, the program should be compiled with nvcc using the architecture specification that matches your GPU. 
Replace sm_cc with the compute capability of your GPU.

```
nvcc -arch=sm_cc sinreduction.cu -o sinreduction
```

## Current Status
The CPU version gives correct answer with current precision. For GPU, the abnormal depedence to the input is happended.

For my RTX3080(8G) in WSL2 subsystem(cuda 12.2), result is correct with first agrument is 65600 and second one has no dependence.

Need more observation to it.
