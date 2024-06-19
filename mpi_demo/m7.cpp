#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

float* createRdNum(int num_ele)
{
    float* rdn = (float*)malloc(sizeof(float)*num_ele);
    for (int i=0;i < num_ele; i++) {
        rdn[i] = (rand()/(float)RAND_MAX);
    }
    return rdn;
}

int main(int argc, char* argv[])
{
    int num_ele_local_array = atoi(argv[1]);
    int rank, np;
    float local_sum=0;
    float global_sum=0;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    srand(time_t(NULL)*rank);
    float* rdn = createRdNum(num_ele_local_array);

    // sum
    for (int i = 0;i < num_ele_local_array; i++) {
        local_sum += rdn[i];
    }
    printf("local sum for process %d = %f, avg = %f\n", rank, local_sum, local_sum/num_ele_local_array);

    MPI_Reduce(
        &local_sum,
        &global_sum,
        1,
        MPI_FLOAT,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        printf("global sum = %f, avg = %f\n", 
            global_sum,
            global_sum / (num_ele_local_array * np)
        );
    }

    free(rdn);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}