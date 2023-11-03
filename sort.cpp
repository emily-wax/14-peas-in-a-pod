void check_sort(int* thread_values_array, int block_size){
    int num_threads;
    int thread_id;
    MPI_Comm_size(MPI_COMM_WORLD,&num_threads);
    MPI_Comm_rank(MPI_COMM_WORLD, &thread_id);

    // each thread is going to check itself to see if it's sorted
    for(int i = 0; i < block_size-1; i++){
        if (thread_values_array[i] > thread_values_array[i+1]){
            cout << "ERROR NOT SORTED\n" << endl;
        }
    }

    // thread sends its [0] to id-1
    if (thread_id > 0)
        MPI_Send(&thread_values_array[0], 1, MPI_INT, thread_id-1, 0, MPI_COMM_WORLD);

    // thread sends its [-1] to id+1
    // check if index is correct 
    if (thread_id < num_threads-1)
        MPI_Send(&thread_values_array[block_size-1], 1, MPI_INT, thread_id+1, 0, MPI_COMM_WORLD);

    // thread receives id-1's [-1] and checks to see if it's < its [0]
    int above, below;
    if (thread_id > 0){
        MPI_Recv(&below, 1, MPI_INT, thread_id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (below > thread_values_array[0]){
            cout << "ERROR NOT SORTED\n" << endl;
        }
    }

    // thread receives id+1's [0] and checks to see if it's > its [-1]
    if (thread_id < num_threads-1){
        MPI_Recv(&above, 1, MPI_INT, thread_id + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (above < thread_values_array[0]){
            cout << "ERROR NOT SORTED\n" << endl;
        }
    }
}
