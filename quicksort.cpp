/******************************************************************************
* FILE: quicksort.cpp
* DESCRIPTION:  
*   This file is the start of the quicksort algorithm using scatter/merge
* AUTHOR: Emily Wax
* LAST REVISED: 11/01/23
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <iostream>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include "data_creation.cpp"

/*
Helpful MPI calls:
    - MPI_send
    - MPI_recv

    -MPI_Probe & MPI_Get_Count - can get size of array that's coming in receive

*/

// TODO: implement swap function

/* pass in indices */
void swap(int* arr, int x, int y)
{
    int temp = arr[x];
    arr[x] = arr[y];
    arr[y] = temp;
}


// TODO: implement quicksort function
void quicksort(int* arr, int start_index, int end_index)
{
    // Index keeps track of pivot's real spot
    int pivot, index;

    // If size is 1, return
    if( (start_index - end_index) < 1)
        return;

    // Pick pivot and move it to the front
    pivot = arr[(start_index + end_index)/ 2];
    swap( arr, end_index, (start_index + end_index) / 2);

    // Iterate through array and swap to divide into low and high
    // last index is skipped because it is pivot. 
    index = start_index - 1;
    for( int i = start_index + 1; i < end_index; i++ )
    {
        if(arr[i] < pivot)
        {
            index++;
            swap(arr, index, i);
        }
    }

    // Put pivot back into correct spot
    // incrementing index one more because we put pivot at the back
    index++;
    swap(arr, index, end_index);

    // Recursive call to quicksort
    quicksort(arr, start_index, index - start_index);
    quicksort(arr, index + 1, end_index);

}


int main(int argc, char* argv[]){
    int NUM_VALS = atoi(argv[1]);
    int num_threads;
    // nullptr won't matter to worker threads
    int* values_array = nullptr;
    int* sub_array;
    int* chunk_received;
    int thread_id;
    int chunk_size, received_size;
    MPI_Status status;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&num_threads);
    MPI_Comm_rank(MPI_COMM_WORLD, &thread_id);

    createData(num_threads, values_array, NUM_VALS, SORTED);

    MPI_Barrier(MPI_COMM_WORLD);

    // Broadcast number of elements (may not need to do this)
    // MPI_Bcast(&NUM_VALS)

    // Compute Chunk Size - cover if it isn't evenly divisible maybe
    chunk_size = NUM_VALS / num_threads;
    sub_array = (int*)malloc(chunk_size * sizeof(int));

    // Scatter data
    MPI_Scatter(values_array, chunk_size, MPI_INT, sub_array, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    free(values_array);
    values_array = NULL;
    // free array?

    // have each chunk quicksort independently
    quicksort(sub_array, 0, chunk_size - 1);

    // use send and receive to collect and merge data
    for(int step = 1; step < num_threads; step *= 2)
    {
        if(thread_id % (2 * step) != 0)
        {
            MPI_Send(sub_array, chunk_size, MPI_INT, thread_id - step, 0, MPI_COMM_WORLD);
            break;
        }

        

        if(thread_id + step < num_threads)
        {
            if ( NUM_VALS >= chunk_size * (thread_id + 2 * step))
            {
                received_size = chunk_size * step;
            } else
            {
                received_size = NUM_VALS - chunk_size * (thread_id + step);
            }

            chunk_received = (int*)malloc(received_size * sizeof(int));
            
            MPI_Recv(chunk_received, received_size, MPI_INT, thread_id + step, 0, MPI_COMM_WORLD, &status);

            values_array = 
        }
    }

    if (thread_id == 0){
        delete[] values_array;
    }
    MPI_Finalize();
    
}