#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char* argv[])
{
    int rank;
    int tag = 81;
    char msg[20];
    int sent_result;
    int received_result;
    MPI_Init(&argc, &argv);
    MPI_Status status;
    strcpy(msg, "Gold Coast");
    

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        sent_result = MPI_Send(
            msg,
            strlen(msg)+1,
            MPI_CHAR,
            1,
            tag,
            MPI_COMM_WORLD
        );
        if (sent_result != MPI_SUCCESS) {
            fprintf(stderr, "MPI_Send failed with err code %d ", sent_result);
            MPI_Abort(MPI_COMM_WORLD, 1);
        } else {
            printf("Process 0 send message - %s - to Process 1\n", msg);
        }
    } else if (rank==1) {
        received_result = MPI_Recv(
            msg,
            strlen(msg)+1,
            MPI_CHAR,
            0,
            tag,
            MPI_COMM_WORLD,
            &status
        );
        if (received_result != MPI_SUCCESS) {
            fprintf(stderr, "MPI_Recv failed with err code %d ", received_result);
            MPI_Abort(MPI_COMM_WORLD, 1);
        } else {
            int received_size;
            MPI_Get_count(
                &status, MPI_CHAR, &received_size);
            printf("Process 1 received message - %s - with size %d from Process 0\n",msg,received_size);
        }
    }


    MPI_Finalize();
    return 0;
}