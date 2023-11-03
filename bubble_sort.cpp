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
            sequential_bubble(thread_values_array, 0, block_size);
            // printArrayAll(thread_values_array, block_size);
            if (thread_id % 2 == 0){
                // recieve block end_index to end_index + block size, from thread_id 1 greater than self
                MPI_Recv(&thread_values_array[block_size], block_size, MPI_INT, thread_id + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // printArrayAll(thread_values_array, block_size);
                sequential_bubble(thread_values_array, 0,  2 * block_size);
                // printArrayAll(thread_values_array, block_size);
                // send second half of list (end_index to end_index + block size) to thread_id 1 greater than self
                MPI_Send(&thread_values_array[block_size], block_size, MPI_INT, thread_id+1, 0, MPI_COMM_WORLD);
            }
            else{
                
                //send block start_index to end index to thread_id 1 less than self
                MPI_Send(thread_values_array, block_size, MPI_INT, thread_id-1, 0, MPI_COMM_WORLD);
                //recieve new block start_index to end index from thread_id 1 less than self
                MPI_Recv(thread_values_array, block_size, MPI_INT, thread_id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            step = !step;
        }
        else{
            //printArrayAll(thread_values_array, block_size);
            sequential_bubble(thread_values_array, 0, block_size);
            //printArrayAll(thread_values_array, block_size);
            if ((thread_id % 2 == 1) && (thread_id != num_threads -1)){ // not the first or last thread because those would cause the program to hang
                // recieve block end_index to end_index + block size, from thread_id 1 greater than self
                //printArrayAll(thread_values_array, block_size);
                MPI_Recv(&thread_values_array[block_size], block_size, MPI_INT, thread_id + 1, 0, MPI_COMM_WORLD, &statusRecv);
                //cout << "thread id in the top loop is: " << thread_id <<" source is: " << statusRecv.MPI_SOURCE << endl;
                //printArrayAll(thread_values_array, block_size);
                sequential_bubble(thread_values_array, 0, 2 * block_size);
                //printArrayAll(thread_values_array, block_size);
                // send second half of list (end_index to end_index + block size) to thread_id 1 greater than self
                MPI_Send(&thread_values_array[block_size], block_size, MPI_INT, thread_id+1, 0, MPI_COMM_WORLD);
            }
            else if ((thread_id % 2 == 0) && (thread_id != 0)){
                //cout << "thread id in the bottom loop: " << thread_id << endl;
                //send block start_index to end index to thread_id 1 less than self
                MPI_Send(thread_values_array, block_size, MPI_INT, thread_id-1, 0, MPI_COMM_WORLD);
                //recieve new block start_index to end index from thread_id 1 less than self
                MPI_Recv(thread_values_array, block_size, MPI_INT, thread_id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
    // nullptr won't matter to worker threads
    
    int thread_id;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&num_threads);
    MPI_Comm_rank(MPI_COMM_WORLD, &thread_id);

    int block_size = NUM_VALS / num_threads;
    int* thread_values_array = (int*) malloc (block_size * 2 * sizeof(int));

    //cout << "Size is: " << num_threads << " Block Size is: " << block_size << endl;

    fillArray(thread_values_array, block_size, NUM_VALS, PERTURBED);

    bubble_sort(thread_values_array, NUM_VALS);


    printArrayTogether(thread_values_array, block_size);

    check_sort(thread_values_array, block_size); 

    MPI_Finalize();
}


