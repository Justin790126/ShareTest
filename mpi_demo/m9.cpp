#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank,np;
    int a;
    double b;
    char packbuff[100];
    int packedsize, retrieve_pos;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&np);

    do
    {
        packedsize = 0;
        if (rank == 0) {
            printf("Enter an integer and a double: ");
            scanf("%d %lf", &a, &b);
            MPI_Pack(
                &a,
                1,
                MPI_INT,
                packbuff,
                100,
                &packedsize,
                MPI_COMM_WORLD
            );
            MPI_Pack(
                &b,
                1,
                MPI_DOUBLE,
                packbuff,
                100,
                &packedsize,
                MPI_COMM_WORLD
            );
        }

    
        MPI_Bcast(
            &packedsize,
            1,
            MPI_INT,
            0,
            MPI_COMM_WORLD);
        MPI_Bcast(
            packbuff,
            packedsize,
            MPI_PACKED,
            0,
            MPI_COMM_WORLD);
        
        if (rank != 0) {
            retrieve_pos = 0;
            MPI_Unpack(
                packbuff,
                packedsize,
                &retrieve_pos,
                &a,
                1,
                MPI_INT,
                MPI_COMM_WORLD
            );
            MPI_Unpack(
                packbuff,
                packedsize,
                &retrieve_pos,
                &b,
                1,
                MPI_DOUBLE,
                MPI_COMM_WORLD
            );
        }

        printf("Proces %d got %d and %lf\n",
            rank,
            a,
            b);
    } while (a>= 0 && b>= 0);

    MPI_Finalize();

    return 0;
}