#include <stdio.h>      // Printf
#include <time.h>       // Timer
#include <math.h>       // Logarithm
#include <stdlib.h>     // Malloc
#include "mpi.h"        // MPI Library
#include <limits.h>
#include <iostream>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define MASTER 0        // Who should do the final processing?
#define OUTPUT_NUM 10   // Number of elements to display in output


// Globals
// Not ideal for them to be here though
double timer_start;
double timer_end;
int process_rank;
int num_processes;
int * array;
int block_size;

enum sort_type{
  SORTED,
  REVERSE_SORTED,
  PERTURBED,
  RANDOM
};

void printArray(int* values, int num_values, int thread_id){
    std::cout << "for Thread_id: " << thread_id << "\nArray is: \n";
    for (int i = 0; i < num_values; i++){
        std::cout << values[i] << ", ";
    }

    std::cout << std::endl << std::endl;
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
                    std::cout << thread_values_array[i];
                }
                else{
                    std::cout << thread_values_array[i] << ", ";
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    std::cout << std::endl;
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

// generate data based on input
void generate_data(int* values, int block_size, int NUM_VALS, int sort_type)
{
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


///////////////////////////////////////////////////
// Comparison Function
///////////////////////////////////////////////////
int ComparisonFunc(const void * a, const void * b) {
    return ( * (int *)a - * (int *)b );
}

///////////////////////////////////////////////////
// Compare Low
///////////////////////////////////////////////////
void CompareLow(int j) {
    int i, min;

    /* Sends the biggest of the list and receive the smallest of the list */

    // Send entire array to paired H Process
    // Exchange with a neighbor whose (d-bit binary) processor number differs only at the jth bit.
    int send_counter = 0;
    int * buffer_send = (int*) malloc((block_size + 1) * sizeof(int));
    MPI_Send(
        &array[block_size - 1],     // entire array
        1,                          // one data item
        MPI_INT,                    // INT
        process_rank ^ (1 << j),    // paired process calc by XOR with 1 shifted left j positions
        0,                          // tag 0
        MPI_COMM_WORLD              // default comm.
    );

    // Receive new min of sorted numbers
    int recv_counter;
    int * buffer_recieve = (int*) malloc((block_size + 1) * sizeof(int));
    MPI_Recv(
        &min,                       // buffer the message
        1,                          // one data item
        MPI_INT,                    // INT
        process_rank ^ (1 << j),    // paired process calc by XOR with 1 shifted left j positions
        0,                          // tag 0
        MPI_COMM_WORLD,             // default comm.
        MPI_STATUS_IGNORE           // ignore info about message received
    );

    // Buffers all values which are greater than min send from H Process.
    for (i = 0; i < block_size; i++) {
        if (array[i] > min) {
            buffer_send[send_counter + 1] = array[i];
            send_counter++;
        } else {
            break;      // Important! Saves lots of cycles!
        }
    }

    buffer_send[0] = send_counter;

    // send partition to paired H process
    MPI_Send(
        buffer_send,                // Send values that are greater than min
        send_counter,               // # of items sent
        MPI_INT,                    // INT
        process_rank ^ (1 << j),    // paired process calc by XOR with 1 shifted left j positions
        0,                          // tag 0
        MPI_COMM_WORLD              // default comm.
    );

    // receive info from paired H process
    MPI_Recv(
        buffer_recieve,             // buffer the message
        block_size,                 // whole array
        MPI_INT,                    // INT
        process_rank ^ (1 << j),    // paired process calc by XOR with 1 shifted left j positions
        0,                          // tag 0
        MPI_COMM_WORLD,             // default comm.
        MPI_STATUS_IGNORE           // ignore info about message received
    );

    // Take received buffer of values from H Process which are smaller than current max
    for (i = 1; i < buffer_recieve[0] + 1; i++) {
        if (array[block_size - 1] < buffer_recieve[i]) {
            // Store value from message
            array[block_size - 1] = buffer_recieve[i];
        } else {
            break;      // Important! Saves lots of cycles!
        }
    }

    // Sequential Sort
    sequential_bubble(array, 0, block_size);

    // Reset the state of the heap from Malloc
    free(buffer_send);
    free(buffer_recieve);

    return;
}


///////////////////////////////////////////////////
// Compare High
///////////////////////////////////////////////////
void CompareHigh(int j) {
    int i, max;

    // Receive max from L Process's entire array
    int recv_counter;
    int * buffer_recieve = (int*) malloc((block_size + 1) * sizeof(int));
    MPI_Recv(
        &max,                       // buffer max value
        1,                          // one item
        MPI_INT,                    // INT
        process_rank ^ (1 << j),    // paired process calc by XOR with 1 shifted left j positions
        0,                          // tag 0
        MPI_COMM_WORLD,             // default comm.
        MPI_STATUS_IGNORE           // ignore info about message received
    );

    // Send min to L Process of current process's array
    int send_counter = 0;
    int * buffer_send = (int*) malloc((block_size + 1) * sizeof(int));
    MPI_Send(
        &array[0],                  // send min
        1,                          // one item
        MPI_INT,                    // INT
        process_rank ^ (1 << j),    // paired process calc by XOR with 1 shifted left j positions
        0,                          // tag 0
        MPI_COMM_WORLD              // default comm.
    );

    // Buffer a list of values which are smaller than max value
    for (i = 0; i < block_size; i++) {
        if (array[i] < max) {
            buffer_send[send_counter + 1] = array[i];
            send_counter++;
        } else {
            break;      // Important! Saves lots of cycles!
        }
    }

    // Receive blocks greater than min from paired slave
    MPI_Recv(
        buffer_recieve,             // buffer message
        block_size,                 // whole array
        MPI_INT,                    // INT
        process_rank ^ (1 << j),    // paired process calc by XOR with 1 shifted left j positions
        0,                          // tag 0
        MPI_COMM_WORLD,             // default comm.
        MPI_STATUS_IGNORE           // ignore info about message receiveds
    );
    recv_counter = buffer_recieve[0];

    // send partition to paired slave
    buffer_send[0] = send_counter;
    MPI_Send(
        buffer_send,                // all items smaller than max value
        send_counter,               // # of values smaller than max
        MPI_INT,                    // INT
        process_rank ^ (1 << j),    // paired process calc by XOR with 1 shifted left j positions
        0,                          // tag 0
        MPI_COMM_WORLD              // default comm.
    );

    // Take received buffer of values from L Process which are greater than current min
    for (i = 1; i < recv_counter + 1; i++) {
        if (buffer_recieve[i] > array[0]) {
            // Store value from message
            array[0] = buffer_recieve[i];
        } else {
            break;      // Important! Saves lots of cycles!
        }
    }

    // Sequential Sort
    sequential_bubble(array, 0, block_size);

    // Reset the state of the heap from Malloc
    free(buffer_send);
    free(buffer_recieve);

    return;
}



///////////////////////////////////////////////////
///////////////////////////////////////////////////
int main(int argc, char * argv[]) {
    int i, j;

    // Initialization, get # of processes & this PID/rank
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    int num_vals = atoi(argv[1]); 

    // Initialize Array for Storing Random Numbers
    block_size = num_vals / num_processes;
    
    array = (int *) malloc(block_size * sizeof(int));

    generate_data(array, block_size, num_vals, REVERSE_SORTED);

    printArrayTogether(array, block_size);

    // Blocks until all processes have finished generating
    MPI_Barrier(MPI_COMM_WORLD);

    // Begin Parallel Bitonic Sort Algorithm from Assignment Supplement

    // Cube Dimension
    int dimensions = (int)(log2(num_processes));

    // Start Timer before starting first sort operation (first iteration)
    if (process_rank == MASTER) {
        printf("Number of Processes spawned: %d\n", num_processes);
        timer_start = MPI_Wtime();
    }

    // Sequential Sort
    sequential_bubble(array, 0, block_size);

    // Bitonic Sort follows
    for (i = 0; i < dimensions; i++) {
        for (j = i; j >= 0; j--) {
            // (window_id is even AND jth bit of process is 0)
            // OR (window_id is odd AND jth bit of process is 1)
            if (((process_rank >> (i + 1)) % 2 == 0 && (process_rank >> j) % 2 == 0) || ((process_rank >> (i + 1)) % 2 != 0 && (process_rank >> j) % 2 != 0)) {
                CompareLow(j);
            } else {
                CompareHigh(j);
            }
        }
    }

    // Blocks until all processes have finished sorting
    MPI_Barrier(MPI_COMM_WORLD);

    // Reset the state of the heap from Malloc
    free(array);

    printArrayTogether(array, block_size);
    // Done
    MPI_Finalize();

    return 0;
}
