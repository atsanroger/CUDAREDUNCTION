#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>
#include "cxtimers.h"


inline float sinsum(float x, int terms){

    float term = x;
    float sum  = term;
    float x2   = x*x;

    for(int n = 1; n < terms ; n++){
        term *= -x2 / (float)(2*n*(2*n+1));
        sum += term;
    }
    return sum;
}

int main(int argc, char *argv[]){

    int    rank, size;
    double global_sum = 0.0;

    MPI_Init(&argc, &argv); // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get current process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get number of processes

    int steps = (argc >1) ? atoi(argv[1]) : 10000000;
    int terms = (argc >2) ? atoi(argv[2]) : 1000;

    double pi = M_PI;
    double step_size = pi / (steps-1);

    // Divide steps across processes
    int local_steps = steps / size;
    int start_step = rank * local_steps;
    int end_step = (rank == size-1) ? steps : start_step + local_steps;

    cx::timer tim;
    double local_sum = 0.0;

    #pragma omp parallel for reduction(+:local_sum)
    for(int step = start_step; step < end_step ; step++){
        float x = step_size * step;
        local_sum += sinsum(x, terms);
    }

    // Gather all partial results
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        double cpu_time = tim.lap_ms();
        // Trapezoidal Rule correction
        global_sum -= 0.5*(sinsum(0.0, terms)+sinsum(pi, terms));
        global_sum *= step_size; 
        printf("cpu sum = %.10f, steps %d terms %d time %.3f ms\n", global_sum , steps , terms , cpu_time);
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}
