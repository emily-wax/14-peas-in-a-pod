bool check_sort(int* thread_values_array, int block_size){
    bool sorted = true;
    // each thread is going to check itself to see if it's sorted
    for(int i = 0; i < block_size-1; i++)
        sorted = sorted && (thread_values_array[i] < thread_values_array[i+1]);

    // thread sends its [0] to id-1
    MPI_Send(thread_values_array[0], 1, MPI_INT, thread_id-1, 0, MPI_COMM_WORLD);

    // thread sends its [-1] to id+1
    // check if index is correct 
    MPI_Send(thread_values_array[block_size-1], 1, MPI_INT, thread_id+1, 0, MPI_COMM_WORLD);

    // thread receives id-1's [-1] and checks to see if it's < its [0]
    int* above, below;
    MPI_Recv(&below, 1, MPI_INT, thread_id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    sorted = sorted && (&below < thread_values_array[0]);

    // thread receives id+1's [0] and checks to see if it's > its [-1]
    MPI_Recv(&above, 1, MPI_INT, thread_id + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    sorted = sorted && (&above > thread_values_array[block_size-1]);

    return sorted;
}