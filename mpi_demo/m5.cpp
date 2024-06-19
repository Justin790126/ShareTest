#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 100

int main(int argc, char* argv[])
{
    int id,np;

    int global_array[N];
    int local_array[N/4];
    int local_sum = 0;
    int global_sum = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    if (id == 0) {
        // init array with 1, 2, 3 ... 100
        for (int i = 0;i < N; ++i) {
            global_array[i] = i + 1;
        }
    }

    // Scatter global array to local array
    MPI_Scatter(
        global_array,
        N/np,
        MPI_INT,
        local_array,
        N/np,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // calculate local sum
    for (int i = 0; i < N/np; ++i)
    {
        local_sum += local_array[i];
    }

    // Gather local sum
    MPI_Gather(
        &local_sum,
        1,
        MPI_INT,
        global_array,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // global_array = [sum1, sum2, sum3 ...]
    if (id == 0) {
        for (size_t i = 0; i < np; i++)
        {
            global_sum += global_array[i];
        }
        printf("Global sum: %d\n", global_sum);
    }



    MPI_Finalize();
    return 0;
}