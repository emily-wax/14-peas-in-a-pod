 /******************************************************************************
* FILE: sample_sort.c
* DESCRIPTION:  
*   Sample sort MPI implementation
* AUTHOR: Emily Wax
* LAST REVISED: 11/2/23
******************************************************************************/
#include "mpi.h"
#include <string>

#include <algorithm>
//#include <compare>
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

/* Define Caliper Region Names*/
const char* main_cali = "main"; 
const char* data_init = "data_init";
const char* correctness_check = "correctness_check"; 
const char* comm = "comm"; 
const char* comm_small = "comm_small"; 
const char* comm_large = "comm_large"; 
const char* comp = "comp";
const char* comp_small = "comp_small"; 
const char* comp_large = "comp_large"; 
const char* broadcast = "MPI_Bcast";
const char* gather = "MPI_Gather";
const char* send   = "MPI_Send";
const char* recv   = "MPI_Recv";

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

/* Function to print a process's value array */
void printArray(vector<int> values, int num_values, int proc_id){
    cout << "for Thread_id: " << proc_id << "\nArray is: \n";
    for (int i = 0; i < num_values; i++){
        cout << values[i] << ", ";
    }

    cout << endl << endl;
}

/* check sort - compares each processes values to make sure it's sorted
                and is sorted in comparison to others. */
bool check_sort(vector<int> thread_values_array, int size){

    // each thread is going to check itself to see if it's sorted
    for(int i = 0; i < size-1; i++){
        if (thread_values_array[i] > thread_values_array[i+1]){
            return false;
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
            return false;
        }
    }

    // thread receives id+1's [0] and checks to see if it's > its [-1]
    if (proc_id < num_procs-1){
        MPI_Recv(&above, 1, MPI_INT, proc_id + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (above < thread_values_array.back()){
            return false;
        }
    }

    return true;
}

/* choose_splitters - chooses p - 1 local splitters. Root process gathers
                       all process splittesr, sorts, and chooses global
                       splitters */
void choose_splitters(int* local_splitter, int* values, int num_vals){
    int* global_splitter;

    // Choose p-1 local splitters (evenly separated)
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);

    for(int i = 0; i < (num_procs - 1); i++)
    {
        local_splitter[i] = values[num_vals/(num_procs * num_procs) * (i+1)];
    }

    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    // Send local splitters back to root process
    global_splitter = (int *)malloc( sizeof(int) * num_procs * (num_procs - 1));

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    CALI_MARK_BEGIN(gather);

    MPI_Gather(local_splitter, num_procs - 1, MPI_INT, global_splitter, num_procs - 1, MPI_INT, 0, MPI_COMM_WORLD);

    CALI_MARK_END(gather);
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);

    // Root: run sort on splitters array
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    if(proc_id == 0)
    {

        qsort( (char *) global_splitter, num_procs * (num_procs - 1), sizeof(int), intCompare);

        // Choose p-1 Global splitters (evenly separated)
        for(int i = 0; i < num_procs - 1; i++)
        {
            local_splitter[i] = global_splitter[(num_procs - 1) * (i + 1)];
        }
    }
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    MPI_Barrier(MPI_COMM_WORLD);
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

/* sort local process values into buckets, bounds are inclusive on low end, exclusive on high end [) */
void fillBuckets( vector<vector<int>>& buckets, int* global_splitters, int* values, int block_size){

    int curr_val;

    for(int i = 0; i < block_size; i++)
    {
        curr_val = values[i];

        for(int j = 0; j < (num_procs - 1); j++)
        {
            // cover low bounds
            if (curr_val < global_splitters[j])
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
}

// Send values to each process based on its bucket indices (except self)
vector<int> bucketComm(vector<vector<int>> buckets, int block_size){
    
    int* recvbuf = (int *)malloc(sizeof(int) * block_size);
    vector<int> finalBucket;
    int recv_cnt;
    MPI_Status status;

    for(int i = 0; i < num_procs; i++)
    {
        if( i != proc_id )
        {
            CALI_MARK_BEGIN(comm);
            CALI_MARK_BEGIN(comm_large);
            CALI_MARK_BEGIN(recv);

            MPI_Recv(recvbuf, block_size, MPI_INT, i, 0, MPI_COMM_WORLD, &status );

            CALI_MARK_END(recv);
            CALI_MARK_END(comm_large);
            CALI_MARK_END(comm);
            

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

                    CALI_MARK_BEGIN(comm);
                    CALI_MARK_BEGIN(comm_large);
                    CALI_MARK_BEGIN(send);

                    MPI_Send( (buckets[j]).data(), (buckets[j]).size(), MPI_INT, j, 0, MPI_COMM_WORLD);

                    CALI_MARK_END(send);
                    CALI_MARK_END(comm_large);
                    CALI_MARK_END(comm);
                }
            }
        }
    }

    // add own into bucket
    for(int i = 0; i < (buckets[proc_id]).size(); i++)
    {
        finalBucket.push_back(buckets[proc_id][i]);
    }

    return finalBucket;
}

int main(int argc, char* argv[]){

    // Command Line Arguments: size, processes
    int NUM_VALS = atoi(argv[1]);
    int* local_splitter;
    bool sorted;
    int sorttype = atoi(argv[2]);
    
    // MPI initialization 
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD,&num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

    CALI_MARK_BEGIN(main_cali);

    // Divide data into process values
    int block_size = NUM_VALS / num_procs;
    int* proc_values_array = (int*) malloc (block_size * sizeof(int));

    /* Data Generation */
    CALI_MARK_BEGIN(data_init);
    fillArray(proc_values_array, block_size, NUM_VALS, sorttype);
    CALI_MARK_END(data_init);

    local_splitter = (int *) malloc(sizeof(int) * (num_procs-1));

    choose_splitters( local_splitter, proc_values_array, NUM_VALS );

    // Broadcast Global Splitters to all other processes
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    CALI_MARK_BEGIN(broadcast);

    MPI_Bcast( local_splitter, num_procs - 1, MPI_INT, 0, MPI_COMM_WORLD);

    CALI_MARK_END(broadcast);
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);
    
    // Each process:
    // Use global splitters to choose bucket indices
    vector<vector<int>> buckets;

    // create a vector for each bucket (p)
    for(int i = 0; i < num_procs; i++)
    {
        vector<int> vec;
        buckets.push_back(vec);
    }


    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);

    fillBuckets( buckets, local_splitter, proc_values_array, block_size);

    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    // Send values to each process based on its bucket indices (except self)
    vector<int> finalBucket;


    finalBucket = bucketComm( buckets, block_size);


    // Run sort on each process
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);

    sort(finalBucket.begin(), finalBucket.end());

    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);


    // Check if actually sorted
    CALI_MARK_BEGIN(correctness_check);

    sorted = check_sort(finalBucket, finalBucket.size());

    CALI_MARK_END(correctness_check);

    CALI_MARK_END(main_cali);

    string sort_string;
    string check_string;

    if( sorttype == SORTED)
    {
        sort_string = "sorted";
    } else if( sorttype == PERTURBED){
        sort_string = "1 perturbed";
    } else if( sorttype == REVERSE_SORTED){
        sort_string  = "reversed";
    } else{
        sort_string = "random";
    }

    if(sorted){
        check_string = "success";
    } else{
        check_string = "failure";
    }

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "Sample Sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", sort_string); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
    adiak::value("num_threads", 0); // The number of processors (MPI ranks)
    adiak::value("group_num", 14); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    adiak::value("correctness", check_string);


    MPI_Finalize();

}
