 /******************************************************************************
* FILE: sample_sort.c
* DESCRIPTION:  
*   Sample sort MPI implementation
* AUTHOR: Emily Wax
* LAST REVISED: 11/2/23
******************************************************************************/
#include "mpi.h"


#include <algorithm>
#include <compare>
#include <iostream>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

/* global variables */
int num_procs;
int proc_id;

enum sort_type{
    SORTED,
    REVERSE_SORTED,
    PERTURBED,
    RANDOM
};


/* Integer compare function to use with quicksort() */
static int intCompare(const void *i, const void *j){
    if((*(int *)i) > (*(int *)j))
        return 1;
    if((*(int *)i) < (*(int *)j))
        return -1;
    return 0;
}

/* check sort - compares each processes values to make sure it's sorted
                and is sorted in comparison to others. */
void check_sort(vector<int> thread_values_array, int size){

    // each thread is going to check itself to see if it's sorted
    for(int i = 0; i < size-1; i++){
        if (thread_values_array[i] > thread_values_array[i+1]){
            cout << "ERROR NOT SORTED\n" << endl;
        }
    }

    // thread sends its [0] to id-1
    if (proc_id > 0)
        MPI_Send(&thread_values_array[0], 1, MPI_INT, proc_id-1, 0, MPI_COMM_WORLD);

    // thread sends its [-1] to id+1
    // check if index is correct 
    if (proc_id < num_procs-1)
        MPI_Send(&thread_values_array[thread_values_array.size() - 1], 1, MPI_INT, proc_id+1, 0, MPI_COMM_WORLD);

    // thread receives id-1's [-1] and checks to see if it's < its [0]
    int above, below;
    if (proc_id > 0){
        MPI_Recv(&below, 1, MPI_INT, proc_id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (below > thread_values_array[0]){
            cout << "ERROR NOT SORTED\n" << endl;
        }
    }

    // thread receives id+1's [0] and checks to see if it's > its [-1]
    if (proc_id < num_procs-1){
        MPI_Recv(&above, 1, MPI_INT, proc_id + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (above < thread_values_array.back()){
            cout << "ERROR NOT SORTED\n" << endl;
        }
    }

}

/* choose_splitters - chooses p - 1 local splitters. Root process gathers
                       all process splittesr, sorts, and chooses global
                       splitters */
void choose_splitters(int* local_splitter, int* values, int num_vals){
    int* global_splitter;

    // Choose p-1 local splitters (evenly separated)
    for(int i = 0; i < (num_procs - 1); i++)
    {
        local_splitter[i] = values[num_vals/(num_procs * num_procs) * (i+1)];
    }

    // Send local splitters back to root process
    global_splitter = (int *)malloc( sizeof(int) * num_procs * (num_procs - 1));
    MPI_Gather( local_splitter, num_procs - 1, MPI_INT, global_splitter, num_procs - 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Root: run sort on splitters array
    if(proc_id == 0)
    {
        qsort( (char *) global_splitter, num_procs * (num_procs - 1), sizeof(int), intCompare);

        // Choose p-1 Global splitters (evenly separated)
        for(int i = 0; i < num_procs - 1; i++)
        {
            local_splitter[i] = global_splitter[(num_procs - 1) * (i + 1)];
            
        }
    }
}

/* Data Generation functions: generates data per process based on sorting type enum */
void fillArray(int* values, int block_size, int NUM_VALS, int sort_type){ 

    int start_val, end_val;

    if( sort_type == REVERSE_SORTED){
        start_val = ( num_procs - proc_id - 1) * block_size;
        end_val = ((( num_procs - proc_id)) * block_size) - 1;
    } else{
        start_val = proc_id * block_size;
        end_val = ((proc_id + 1) * block_size) - 1;
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


/* Function to print a process's value array */
void printArray(int* values, int num_values, int proc_id){
    cout << "for Thread_id: " << proc_id << "\nArray is: \n";
    for (int i = 0; i < num_values; i++){
        cout << values[i] << ", ";
    }

    cout << endl << endl;
}


int main(int argc, char* argv[]){
    // Command Line Arguments: size, processes
    int NUM_VALS = atoi(argv[1]);
    int curr_val;
    int* local_splitter;
    MPI_Status status;
    
    // MPI initialization 
    MPI_Init(&argc, &argv);
    // EW TODO: make these global
    MPI_Comm_size(MPI_COMM_WORLD,&num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

    // Divide data into process values
    int block_size = NUM_VALS / num_procs;
    int* proc_values_array = (int*) malloc (block_size * sizeof(int));

    /* Data Generation */
    fillArray(proc_values_array, block_size, NUM_VALS, PERTURBED);

    local_splitter = (int *) malloc(sizeof(int) * (num_procs-1));

    choose_splitters( local_splitter, proc_values_array, NUM_VALS );

    // Broadcast Global Splitters to all other processes
    MPI_Bcast( local_splitter, num_procs - 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Each process:
    // Use global splitters to choose bucket indices
    vector<vector<int>> buckets;

    // figure out own bounds .. 

    // create a vector for each bucket (p)
    for(int i = 0; i < num_procs; i++)
    {
        vector<int> vec;
        buckets.push_back(vec);
    }
    // *** sort local process values into buckets
    // space wise: iterate over array and send to different buckets over iteration? (since it's all sorted from low to high)
            // can I send part of an array?

    // bounds are inclusive on low end, exclusive on high end [)
    for(int i = 0; i < block_size; i++)
    {
        curr_val = proc_values_array[i];

        for(int j = 0; j < (num_procs - 1); j++)
        {
            // cover low bounds
            if (curr_val < local_splitter[j])
            {
                (buckets[j]).push_back(curr_val);
                break;
            }

            // cover highest splitter bounds (last iteration)
            if ( j == (num_procs - 2) )
            {
                (buckets[ num_procs - 1 ].push_back(curr_val));
            }
        }
    }

    /* bucket testing */
    // if( proc_id == 1 )
    // {
    //     for(int i = 0; i < num_procs; i++)
    //     {
    //         cout << "bucket " << i << ":"; 
    //         for(int j = 0; j < buckets[i].size(); j++)
    //             cout << " " << buckets[i][j];
    //         cout <<  " " << endl;
    //     }
    // }

    // Send values to each process based on its bucket indices (except self)
    int* recvbuf = (int *)malloc(sizeof(int) * block_size);
    vector<int> finalBucket;
    int recv_cnt;
    for(int i = 0; i < num_procs; i++)
    {
        if( i != proc_id )
        {
            MPI_Recv(recvbuf, block_size, MPI_INT, i, 0, MPI_COMM_WORLD, &status );
            // received item count and add to bucket vector
            MPI_Get_count(&status, MPI_INT, &recv_cnt);
            for(int k = 0; k < recv_cnt; k++)
            {
                finalBucket.push_back(recvbuf[k]);
            }
        }
        else
        {
            for(int j = 0; j < num_procs; j++)
            {
                
                if( j != proc_id)
                {
                    MPI_Send( (buckets[j]).data(), (buckets[j]).size(), MPI_INT, j, 0, MPI_COMM_WORLD);
                }
            }
        }
    }

    // add own into bucket
    for(int i = 0; i < (buckets[proc_id]).size(); i++)
    {
        finalBucket.push_back(buckets[proc_id][i]);
    }

    // Run sort on each process
    sort(finalBucket.begin(), finalBucket.end());

    // Check if actually sorted
    check_sort(finalBucket, finalBucket.size());

    MPI_Finalize();

}
