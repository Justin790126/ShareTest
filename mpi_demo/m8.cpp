#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define DATA_SIZE 100

int main() {
    
    int id, np;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    int local_data[DATA_SIZE/np];

    int* original_data = NULL;
    int* gathered_data = NULL;

    if (id == 0) {
        original_data = (int*) malloc(sizeof(int)*DATA_SIZE);
        for (int i = 0;i < DATA_SIZE; ++i) {
            original_data[i] = i+1;
        }

        gathered_data = (int*) malloc(sizeof(int)*DATA_SIZE);
    }

    // scatter original data to all process
    MPI_Scatter(
        original_data,
        DATA_SIZE/np,
        MPI_INT,
        local_data,
        DATA_SIZE/np,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // apply local transform
    for (int i =0;i < DATA_SIZE/np; i++){
        local_data[i] = local_data[i] * local_data[i];
    }

    // gather local data to root process
    MPI_Gather(
        local_data,
        DATA_SIZE/np,
        MPI_INT,
        gathered_data,
        DATA_SIZE/np,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    if (id == 0) {
        printf("original data: ");
        for (size_t i = 0; i < DATA_SIZE; i++)
        {
            printf("%d ", original_data[i]);
        }
        printf("\n");
        printf("gathered data: ");
        for (size_t i = 0; i < DATA_SIZE; i++) {
            printf("%d ", gathered_data[i]);
        }
        printf("\n");
        free(original_data);
        free(gathered_data);
    }

    MPI_Finalize();
    return 0;
}