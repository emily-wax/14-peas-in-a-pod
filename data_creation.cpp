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

void fillArray(int* values, int start_index, int end_index, int sort_type){ // work each thread will do
    if (sort_type == SORTED || sort_type == PERTURBED){
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
}



void createData(int numThreads, int* values, int numValues, int sortType, int thread_id){
 // call fill array for each thread 
 // total process that the main will call 
    
    int start_index, end_index;
    int block_size = numValues / numThreads;
    start_index = block_size * thread_id;
    end_index = block_size * (thread_id +1) - 1;

    fillArray(values, start_index, end_index, sortType);

}


void printArray(int* values, int num_values, int thread_id){
    cout << "\nArray is: Thread id is: "<< thread_id << "\n";
    for (int i = 0; i < num_values; i++){
        cout << values[i] << ", ";
    }

    cout << endl;
}


int main(int argc, char* argv[]){
    int NUM_VALS = atoi(argv[1]);
    int num_threads;
    int thread_id;
    int *values = (int*) malloc( NUM_VALS * sizeof(int));
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&num_threads);
    MPI_Comm_rank(MPI_COMM_WORLD,&thread_id);

    createData(num_threads, values, NUM_VALS, 0, thread_id);

    if (thread_id == 0){
        printArray(values, NUM_VALS, thread_id);
    }

    MPI_Finalize();
    
}