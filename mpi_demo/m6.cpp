#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[])
{
    int np;
    int rank;
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = 60;

    int* local_array = (int*)malloc(sizeof(int)*N/np);

    unsigned int seed = rank+1;
    srand(seed);
    // fill local array
    for (int i =0; i < N/np; i++) {
        local_array[i] = rand()%100;
    }

    printf("Process %d local arr:", rank);
    for (int i =0;i<N/np;i++) {
        printf("%d ", local_array[i]);
    }
    printf("\n");
    // find local maximum
    int local_max = local_array[0];
    for (int i = 1; i < (N/np); i++)
    {
        if (local_array[i] > local_max) local_max = local_array[i];
    }
    
    int* all_max_values = (int*)malloc(np*sizeof(int));
    MPI_Allgather(
        &local_max,
        1,
        MPI_INT,
        all_max_values,
        1,
        MPI_INT,
        MPI_COMM_WORLD
    );

    printf("Process %d Local max: %d, All max values: ", rank, local_max);
    for (int i = 0; i < np; i++)
    {
        printf("%d ", all_max_values[i]);
        if (i == np-1) printf("\n");
    }
    
    int gl_max = all_max_values[0];
    for (int i = 1; i < np; i++)
    {
        if (all_max_values[i] > gl_max) gl_max = all_max_values[i];
    }
    printf("Process %d, Global max: %d\n", rank, gl_max);
    
    free(local_array);
    free(all_max_values);

    MPI_Finalize();

    return 0;

}