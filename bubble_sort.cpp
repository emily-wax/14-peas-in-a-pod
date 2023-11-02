/******************************************************************************
* FILE: bubble_sort.c
* DESCRIPTION:  
*   This code will be used to sort data using bubble sort
* AUTHOR: Roee Belkin, Ansley Thompson, Emily Wax.
* LAST REVISED: 11/2/23
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <iostream>
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

void printArray(int* values, int num_values){
    cout << "\nArray is: \n";
    for (int i = 0; i < num_values; i++){
        cout << values[i] << ", ";
    }

    cout << endl;
}

// TODO: clean up the code in this function
void fillArray(int* values, int block_size, int NUM_VALS, int sort_type){ // work each thread will do
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



void createData(int numThreads, int* values_array, int NUM_VALS, int sortType){
    int thread_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &thread_id);
    int block_size = NUM_VALS / numThreads;

    if (thread_id == 0){
        values_array = (int*) malloc( NUM_VALS * sizeof(int));
    }
    
    int* thread_values_array = (int*) malloc (block_size * sizeof(int));

    fillArray(thread_values_array, block_size, NUM_VALS, sortType);

    MPI_Gather(thread_values_array, block_size, MPI_INT, values_array, block_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (thread_id == 0)
    {
        printArray(values_array, NUM_VALS);
    }
        
    //delete[] thread_values_array;

}


bool swap(int* values_array, int i, int j){
    if (values_array[i] > values_array[j]){
        int temp = values_array[i];
        values_array[i] = values_array[j];
        values_array[j] = temp;
        return true;
    }
    return false;
}


bool sequential_bubble(int* values_array, int start_index, int end_index){
    bool swapped = false; 
    for (int i = start_index; i < end_index; i++){
        swapped = swap(values_array, i, i+1);
    }
    return swapped; 
}


void bubble_sort(int* values_array, int NUM_VALS){
    int num_threads;
    int thread_id; 
    MPI_Comm_size(MPI_COMM_WORLD,&num_threads);
    MPI_Comm_rank(MPI_COMM_WORLD, &thread_id);

    bool even_swap = true;
    bool odd_swap = true;
    bool step = true; // even vs odd step 
    int block_size = NUM_VALS / num_threads;
    int start_index = block_size * thread_id;
    int end_index = block_size * (thread_id + 1);
    int* thread_values_array = (int*) malloc (block_size * 2 * sizeof(int));
    if (thread_id == 0){
        MPI_Scatter(values_array, block_size, MPI_INT, thread_values_array, block_size, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    while (!even_swap && !odd_swap){
        if (step){
            even_swap = sequential_bubble(thread_values_array, start_index, end_index);
            if (thread_id % 2 == 0){
                // recieve block end_index to end_index + block size, from thread_id 1 greater than self
                MPI_Recv(&thread_values_array[block_size], block_size, MPI_INT, thread_id + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                sequential_bubble(thread_values_array, start_index, end_index + block_size);
                // send second half of list (end_index to end_index + block size) to thread_id 1 greater than self
                MPI_Send(&thread_values_array[block_size], block_size, MPI_INT, thread_id+1, 0, MPI_COMM_WORLD);
            }
            else{
                
                //send block start_index to end index to thread_id 1 less than self
                MPI_Send(thread_values_array, block_size, MPI_INT, thread_id-1, 0, MPI_COMM_WORLD);
                //recieve new block start_index to end index from thread_id 1 less than self
                MPI_Recv(&thread_values_array, block_size, MPI_INT, thread_id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            step = !step;
        }
        else{
            odd_swap = sequential_bubble(thread_values_array, start_index, end_index);
            if (thread_id % 2 == 1 && (thread_id != 0) && thread_id != num_threads -1){ // not the first or last thread because those would cause the program to hang

                // recieve block end_index to end_index + block size, from thread_id 1 greater than self
                MPI_Recv(&thread_values_array[block_size], block_size, MPI_INT, thread_id + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                sequential_bubble(thread_values_array, start_index, end_index + block_size);
                // send second half of list (end_index to end_index + block size) to thread_id 1 greater than self
                MPI_Send(&thread_values_array[block_size], block_size, MPI_INT, thread_id+1, 0, MPI_COMM_WORLD);
            }
            else if (thread_id % 2 == 0){
                //send block start_index to end index to thread_id 1 less than self
                MPI_Send(thread_values_array, block_size, MPI_INT, thread_id-1, 0, MPI_COMM_WORLD);
                //recieve new block start_index to end index from thread_id 1 less than self
                MPI_Recv(&thread_values_array, block_size, MPI_INT, thread_id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            step = !step;
        }
        MPI_Gather(thread_values_array, block_size, MPI_INT, values_array, block_size, MPI_INT, 0, MPI_COMM_WORLD);
        
    }


}




int main(int argc, char* argv[]){
    int NUM_VALS = atoi(argv[1]);
    int num_threads;
    // nullptr won't matter to worker threads
    int* values_array = nullptr;
    int thread_id;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&num_threads);
    MPI_Comm_rank(MPI_COMM_WORLD, &thread_id);

    cout << "Size is: " << num_threads << endl;

    createData(num_threads, values_array, NUM_VALS, PERTURBED);

    //bubble_sort(values_array, NUM_VALS);

    cout << "POST SORT\n\n" << endl;


    if (thread_id == 0){
        printArray(values_array, NUM_VALS); 
        //delete[] values_array;
    }
    MPI_Finalize();
}


