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



const char* comm_small = "comm_small";
const char* comp_small = "comp_small";
const char* comp_medium = "comp_medium";


using namespace std;


enum sort_type{
    SORTED,
    REVERSE_SORTED,
    PERTURBED,
    RANDOM
};

void printArray(int* values, int num_values, int thread_id){
    cout << "for Thread_id: " << thread_id << "\nArray is: \n";
    for (int i = 0; i < num_values; i++){
        cout << values[i] << ", ";
    }

    cout << endl << endl;
}


void printArrayAll(int* thread_values_array, int block_size){
    int num_threads;
    int thread_id;

    MPI_Comm_size(MPI_COMM_WORLD,&num_threads);
    MPI_Comm_rank(MPI_COMM_WORLD, &thread_id);
    for (int i = 0; i < num_threads; i++){
        if (thread_id == i){
            printArray(thread_values_array, block_size, thread_id);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

}


void printArrayTogether(int* thread_values_array, int block_size){
    int num_threads;
    int thread_id;

    MPI_Comm_size(MPI_COMM_WORLD,&num_threads);
    MPI_Comm_rank(MPI_COMM_WORLD, &thread_id);
    for (int i = 0; i < num_threads; i++){
        if (thread_id == i){
            for (int i = 0; i < block_size; i++){
                if ((i == block_size -1) && (thread_id == num_threads -1)){
                    cout << thread_values_array[i];
                }
                else{
                    cout << thread_values_array[i] << ", ";
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
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



bool swap(int* values_array, int i, int j){
    if (values_array[i] > values_array[j]){
        int temp = values_array[i];
        values_array[i] = values_array[j];
        values_array[j] = temp;
        return true;
    }
    return false;
}


void sequential_bubble(int* values_array, int start_index, int end_index){
    bool run = true; 
    while(run){
        run = false; 
        for (int i = start_index; i < end_index -1; i++){
            if (swap(values_array, i, i+1)){
                run = true; 
            }
        }
    }
}


void bubble_sort(int* thread_values_array, int NUM_VALS){
    int num_threads;
    int thread_id; 
    MPI_Status statusSend, statusRecv; 
    MPI_Comm_size(MPI_COMM_WORLD,&num_threads);
    MPI_Comm_rank(MPI_COMM_WORLD, &thread_id);

    bool step = true; // even vs odd step 
    int block_size = NUM_VALS / num_threads;
    
    for (int k = 0; k < num_threads; k++){
        if (step){
            // printArrayAll(thread_values_array, block_size);
            CALI_MARK_BEGIN(comp_small); 
            sequential_bubble(thread_values_array, 0, block_size);
            CALI_MARK_END(comp_small); 
            // printArrayAll(thread_values_array, block_size);
            if (thread_id % 2 == 0){
                // recieve block end_index to end_index + block size, from thread_id 1 greater than self
                CALI_MARK_BEGIN(comm_small);
                MPI_Recv(&thread_values_array[block_size], block_size, MPI_INT, thread_id + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                CALI_MARK_END(comm_small);
                // printArrayAll(thread_values_array, block_size);
                CALI_MARK_BEGIN(comp_medium);
                sequential_bubble(thread_values_array, 0,  2 * block_size);
                CALI_MARK_END(comp_medium);
                // printArrayAll(thread_values_array, block_size);
                // send second half of list (end_index to end_index + block size) to thread_id 1 greater than self
                CALI_MARK_BEGIN(comm_small);
                MPI_Send(&thread_values_array[block_size], block_size, MPI_INT, thread_id+1, 0, MPI_COMM_WORLD);
                CALI_MARK_END(comm_small);
            }
            else{
                
                //send block start_index to end index to thread_id 1 less than self
                CALI_MARK_BEGIN(comm_small);
                MPI_Send(thread_values_array, block_size, MPI_INT, thread_id-1, 0, MPI_COMM_WORLD);
                //recieve new block start_index to end index from thread_id 1 less than self
                MPI_Recv(thread_values_array, block_size, MPI_INT, thread_id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                CALI_MARK_END(comm_small);
            }
            step = !step;
        }
        else{
            //printArrayAll(thread_values_array, block_size);
            CALI_MARK_BEGIN(comp_small);
            sequential_bubble(thread_values_array, 0, block_size);
            CALI_MARK_END(comp_small);
            //printArrayAll(thread_values_array, block_size);
            if ((thread_id % 2 == 1) && (thread_id != num_threads -1)){ // not the first or last thread because those would cause the program to hang
                // recieve block end_index to end_index + block size, from thread_id 1 greater than self
                //printArrayAll(thread_values_array, block_size);
                CALI_MARK_BEGIN(comm_small);
                MPI_Recv(&thread_values_array[block_size], block_size, MPI_INT, thread_id + 1, 0, MPI_COMM_WORLD, &statusRecv);
                CALI_MARK_END(comm_small);
                //cout << "thread id in the top loop is: " << thread_id <<" source is: " << statusRecv.MPI_SOURCE << endl;
                //printArrayAll(thread_values_array, block_size);
                CALI_MARK_BEGIN(comp_medium);
                sequential_bubble(thread_values_array, 0, 2 * block_size);
                CALI_MARK_END(comp_medium);
                //printArrayAll(thread_values_array, block_size);
                // send second half of list (end_index to end_index + block size) to thread_id 1 greater than self
                CALI_MARK_BEGIN(comm_small);
                MPI_Send(&thread_values_array[block_size], block_size, MPI_INT, thread_id+1, 0, MPI_COMM_WORLD);
                CALI_MARK_END(comm_small); 
            }
            else if ((thread_id % 2 == 0) && (thread_id != 0)){
                //cout << "thread id in the bottom loop: " << thread_id << endl;
                //send block start_index to end index to thread_id 1 less than self
                CALI_MARK_BEGIN(comm_small);
                MPI_Send(thread_values_array, block_size, MPI_INT, thread_id-1, 0, MPI_COMM_WORLD);
                //recieve new block start_index to end index from thread_id 1 less than self
                MPI_Recv(thread_values_array, block_size, MPI_INT, thread_id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                CALI_MARK_END(comm_small); 
            }
            step = !step;
        }
        
    }


}

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




int main(int argc, char* argv[]){
    int NUM_VALS = atoi(argv[1]);
    int num_threads;
    int data_to_sort_type = PERTURBED; 
    // nullptr won't matter to worker threads
    
    int thread_id;

    const char* whole_sort = "whole_sort";
    const char* data_creation = "data_creation";
    const char* correctness_check = "correctness_check";
    const char* print_time = "print_time";
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&num_threads);
    MPI_Comm_rank(MPI_COMM_WORLD, &thread_id);

    int block_size = NUM_VALS / num_threads;

    cali::ConfigManager mgr;
    mgr.start();

    int* thread_values_array = (int*) malloc (block_size * 2 * sizeof(int));

    //cout << "Size is: " << num_threads << " Block Size is: " << block_size << endl;

    CALI_MARK_BEGIN(data_creation);
    fillArray(thread_values_array, block_size, NUM_VALS, data_to_sort_type);
    CALI_MARK_END(data_creation);

    CALI_MARK_BEGIN(whole_sort);
    bubble_sort(thread_values_array, NUM_VALS);
    CALI_MARK_END(whole_sort);


    CALI_MARK_BEGIN(print_time);
    printArrayTogether(thread_values_array, block_size);
    CALI_MARK_END(print_time);

    CALI_MARK_BEGIN(correctness_check);
    check_sort(thread_values_array, block_size);
    CALI_MARK_END(correctness_check); 

    if (thread_id == 0){
        adiak::init(NULL);
        adiak::launchdate();    // launch date of the job
        adiak::libraries();     // Libraries used
        adiak::cmdline();       // Command line used to launch the job
        adiak::clustername();   // Name of the cluster
        adiak::value("Algorithm", "Bubble Sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
        adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
        adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
        adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
        adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
        adiak::value("InputType", data_to_sort_type); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
        adiak::value("num_procs", num_threads); // The number of processors (MPI ranks)
        adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
        adiak::value("group_num", 14); // The number of your group (integer, e.g., 1, 10)
        adiak::value("implementation_source", "online + handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    }

    mgr.stop();
    mgr.flush();

    MPI_Finalize();
}


