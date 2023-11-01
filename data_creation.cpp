/******************************************************************************
* FILE: data_creation.c
* DESCRIPTION:  
*   This code will be used to create the 4 different types of data we want to
*   sort on. 
* AUTHOR: Roee Belkin, Ansley Thompson, Emily Wax .
* LAST REVISED: 10/31/23
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

void fillArray(int* values, int block_size, int NUM_VALS, int sort_type){ // work each thread will do
    int thread_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &thread_id);

    int start_index = thread_id * block_size;
    int end_index = ((thread_id + 1) * block_size) -1;
    
    if (sort_type == SORTED){
        int i = start_index;
        while (i <= end_index){
            values[i] = i;
            i++;
        }
    }
    else if (sort_type == REVERSE_SORTED){
        int i = end_index;
        int array_index = start_index;
        while (i >= start_index){
            values[array_index] = i;
            i--;
            array_index++;
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
        while (i <= end_index){
            values[i] = i;
            if (i % 100 == 0){
                values[i] = rand() % (NUM_VALS); 
            }
            i++;
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

    //MPI_Scatter(values_array, block_size, MPI_INT, thread_values_array, block_size, MPI_INT, 0, MPI_COMM_WORLD);

    fillArray(thread_values_array, block_size, NUM_VALS, sortType);

    printArray(thread_values_array, block_size);

    // if (thread_id > 0){
    //     MPI_Gather(thread_values_array, block_size, MPI_INT, nullptr, block_size, MPI_INT, 0, MPI_COMM_WORLD);
    // }

    //if (thread_id == 0){
        MPI_Gather(thread_values_array, block_size, MPI_INT, values_array, block_size, MPI_INT, 0, MPI_COMM_WORLD);
        //printArray(values_array, NUM_VALS);
    //}
    //delete[] thread_values_array;

}




int main(int argc, char* argv[]){
    int NUM_VALS = atoi(argv[1]);
    int num_threads;
    int* values_array = nullptr;
    int thread_id;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&num_threads);
    MPI_Comm_rank(MPI_COMM_WORLD, &thread_id);

    createData(num_threads, values_array, NUM_VALS, SORTED);


    if (thread_id == 0){
        delete[] values_array;
    }
    MPI_Finalize();
    
}