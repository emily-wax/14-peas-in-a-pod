/******************************************************************************
 * FILE: data_creation_cuda.cu
 * DESCRIPTION:
 *   This code will be used to create the 4 different types of data we want to
 *   sort on using CUDA threads.
 * AUTHOR: Roee Belkin, Ansley Thompson, Emily Wax .
 * LAST REVISED: 11/01/23
 ******************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

enum sort_type
{
    SORTED,
    REVERSE_SORTED,
    PERTURBED,
    RANDOM
};

void printArray(int *values, int num_values)
{
    cout << "\nArray is: \n";
    for (int i = 0; i < num_values; i++)
    {
        cout << values[i] << ", ";
    }

    cout << endl;
}

// TODO: clean up the code in this function
void fillArray(int *values, int block_size, int NUM_VALS, int numThreads, int sort_type)
{ // work each thread will do
    int num_threads = numThreads;
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;

    int start_val, end_val;

    if (sort_type == REVERSE_SORTED)
    {
        start_val = (num_threads - thread_id - 1) * block_size;
        end_val = (((num_threads - thread_id)) * block_size) - 1;
    }
    else
    {
        start_val = thread_id * block_size;
        end_val = ((thread_id + 1) * block_size) - 1;
    }

    int start_index = 0;
    int end_index = block_size - 1;

    if (sort_type == SORTED)
    {
        int i = start_index;
        while (i <= end_index)
        {
            values[i] = start_val;
            start_val++;
            i++;
        }
    }
    else if (sort_type == REVERSE_SORTED)
    {
        int i = end_val;
        int array_index = start_index;
        while (i >= start_val)
        {
            values[array_index] = i;
            i--;
            array_index++;
        }
    }
    else if (sort_type == RANDOM)
    {
        int i = start_index;
        while (i <= end_index)
        {
            values[i] = rand() % (INT_MAX);
            i++;
        }
    }
    else if (sort_type == PERTURBED)
    {
        int i = start_index;
        while (i <= end_index)
        {
            values[i] = start_val;
            if (i % 100 == 0)
            {
                values[i] = rand() % (NUM_VALS);
            }
            i++;
            start_val++;
        }
    }
}

void createData(int numThreads, int *values_array, int NUM_VALS, int sortType)
{
    int block_size = NUM_VALS / numThreads;

    values_array = (int *)malloc(NUM_VALS * sizeof(int));

    int *thread_values_array = (int *)malloc(block_size * sizeof(int));

    // may need to initialize block_size and numThreads with dim3 instead of int... we shall see
    // call fillArray with numThreads on block_size data
    // block size may/should be the number of blocks not size

    // sending different parts of values array to different threads to sort, then memcpying all of them back

    fillArray<<<block_size, numThreads>>>(thread_values_array, block_size, NUM_VALS, numThreads, sortType);

    // gather all data somehow

    printArray(values_array, NUM_VALS);
    // delete[] thread_values_array;
}

// take in arguments in the form *.grace_job NUM_VALS NUM_THREADS
int main(int argc, char *argv[])
{
    int NUM_VALS = atoi(argv[1]);
    int NUM_THREADS = atoi(argv[2]);
    // nullptr won't matter to worker threads
    int *values_array = nullptr;
    int thread_id;

    createData(NUM_THREADS, values_array, NUM_VALS, SORTED);

    // if (thread_id == 0)
    // {
    delete[] values_array;
    // }
}