#include <stdio.h>
#include <mpi.h>

int main(){
    MPI_Init(NULL, NULL);

    // Number of processes
    int world_size;
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );

    // Number of current process
    int process_id;
    MPI_Comm_rank( MPI_COMM_WORLD, &process_id );

    // Processor name
    char processor_name[ MPI_MAX_PROCESSOR_NAME ];
    int name_len;
    MPI_Get_processor_name( processor_name, &name_len );

    printf("Hello! - sent from process %d running on processor %s.\n\
        Number of processes is %d.\n\
        Length of proc name is %d.\n\
        ***********************\n",
        process_id, processor_name, world_size, name_len);

    MPI_Finalize();
    return 0;
}
