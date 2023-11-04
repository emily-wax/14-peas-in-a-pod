/******************************************************************************
* FILE: sample_sort.c
* DESCRIPTION:  
*   Sample sort MPI implementation
* AUTHOR: Emily Wax
* LAST REVISED: 11/2/23
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <iostream>
#include <compare>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

enum sort_type{
    SORTED,
    REVERSE_SORTED,
    PERTURBED,
    RANDOM
};

static intcompare(const void *i, const void *j)
{
    if((*(int *)i) > (*(int *)j))
        return 1;
    if((*(int *)i) < (*(int *)j))
        return -1;
    return 0;
}

void printArray(int* values, int num_values, int thread_id){
    cout << "for Thread_id: " << thread_id << "\nArray is: \n";
    for (int i = 0; i < num_values; i++){
        cout << values[i] << ", ";
    }

    cout << endl << endl;
}

// TODO: clean up the code in this function
void fillArray(int* values, int block_size, int NUM_VALS, int sort_type){ 
    int thread_id, num_threads;
    MPI_Comm_rank(MPI_COMM_WORLD, &thread_id);
    MPI_Comm_size(MPI_COMM_WORLD,&num_threads);

    int start_val, end_val;


    if( sort_type == REVERSE_SORTED){
        start_val = ( num_threads - thread_id - 1) * block_size;
        end_val = ((( num_threads - thread_id)) * block_size) - 1;
    } else{
        start_val = thread_id * block_size;
        end_val = ((thread_id + 1) * block_size) - 1;
    }

    int start_index = 0;
    int end_index = block_size - 1;
    
    if (sort_type == SORTED){
        int i = start_index;
        while (i <= end_index){
            values[i] = start_val;
            start_val++;
            i++;
        }
    }
    else if (sort_type == REVERSE_SORTED){
        int i = start_index;
        while (end_val >= start_val){
            values[i] = end_val;
            end_val--;
            i++;
        }
    }
    else if (sort_type == RANDOM){
        int i = start_index; 
        while (i <= end_index){
            values[i] = rand() % (INT_MAX);
            i++;
        }
    }
    else if (sort_type == PERTURBED){
        int i = start_index;
        int perturb_check; 
        while (i <= end_index){
            values[i] = start_val;
            perturb_check = rand() % 100;
            if (perturb_check == 1){
                values[i] = rand() % (NUM_VALS); 
            }
            i++;
            start_val++;
        }

    }
}


int main(int argc, char* argv[]){
    // Command Line Arguments: size, processes
    int NUM_VALS = atoi(argv[1]);
    int num_procs, proc_id;
    int* local_splitter;
    
    // MPI initialization 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

    // Divide data into process values
    int block_size = NUM_VALS / num_procs;
    int* proc_values_array = (int*) malloc (block_size * sizeof(int));

    // Data Generation
    fillArray(proc_values_array, block_size, NUM_VALS, PERTURBED);

    // Have each process sort locally (STL qsort) - optional, probably will not

    // Local Splitters ( p = # processors)

    local_splitter = (int *) malloc (sizeof (int) * (num_procs-1));

        // Choose p-1 local splitters (evenly separated)
    for(int i = 0; i < (num_procs - 1); i++)
    {
        local_splitter[i] = proc_values_array[NUM_VALS/(num_procs * num_procs) * (i+1)]
    }

    // Send local splitters back to root process
    global_splitter = (int *)malloc( sizeof(int) * num_procs * (num_procs - 1))
    MPI_Gather( local_splitter, num_procs - 1, MPI_INT, global_splitter, num_procs - 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Root: run sort on splitters array
    if(proc_id == 0)
    {
        qsort( (char *) global_splitter, num_procs * (num_procs - 1), sizeof(int), intcompare);

        // Choose p-1 Global splitters (evenly separated)
        for(int i = 0; i < num_procs - 1; i++)
        {
            local_splitter[i] = global_splitter[(num_procs - 1) *(i + 1)];
        }
    }

    // Broadcast Global Splitters to all other processes
    MPI_Bcast( local_splitter, num_procs - 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process:
        // Use global splitters to choose bucket indices

        // *** sort local process values into buckets
            // space wise: iterate over array and send to different buckets over iteration? (since it's all sorted from low to high)
                // can I send part of an array?

        // Send values to each process based on its bucket indices

        // create a vector for each processes (use vector of vectors)

    // Run sort on each process

    // Check if actually sorted


}
