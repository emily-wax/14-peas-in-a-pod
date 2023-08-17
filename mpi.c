#include "stdio.h"
#include "mpi.h"

void launch_kernel();

int main(argc, argv)
int argc;
char **argv;
{
   int rank, size;
   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   printf("(MPI) Hello world! Rank = %d, Size = %d.\n", rank, size);
   
   launch_kernel();
   
   MPI_Finalize();
   return 0;
}