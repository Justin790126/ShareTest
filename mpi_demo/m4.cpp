#include <mpi.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char* argv[])
{
    int id, num_process;
    int n,i;
    double h, mypi, pi, sum, x;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_process);
    if (id==0)
    {
        printf("Enter n: ");
        scanf("%d", &n);
    }
    MPI_Bcast(
        &n,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );
    h = 1/(double)n;
    sum = 0.0;
    for (i = id+1; i <=n; i+=num_process) {
        x = h*(double(i)-0.5);
        sum += 4.0/(1+x*x);
    }
    mypi = h*sum;
    MPI_Reduce(
        &mypi,
        &pi,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );
    if (id ==0) {
        printf("pi = %.16f\n", pi);
    }

    MPI_Finalize();
    return 0;
}