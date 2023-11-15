/******************************************************************************
 * FILE: mergesort.cpp
 * DESCRIPTION:
 *   Parallelized merge sort algorithm using CUDA
 * AUTHOR: Harini Kumar
 * LAST REVISED: 11/12/23
 ******************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda_runtime.h>
#include <cuda.h>
#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int THREADS;
int BLOCKS;
int NUM_VALS;

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

void fillArray(int *values, int NUM_VALS, int sort_type)
{
    int i;
    if (sort_type == RANDOM)
    {
        for (i = 0; i < length; ++i)
        {
            arr[i] = rand() % (INT_MAX);
        }
    }
    else if (sort_type == SORTED)
    {
        for (i = 0; i < length; i++)
        {
            arr[i] = i;
        }
    }
    else if (sort_type == PERTURBED)
    {
        for (i = 0; i < length; i++)
        {
            arr[i] = i;
            int temp = rand() % 100;
            if (temp == 1)
            {
                arr[i] = rand() % length;
            }
        }
    }
    else if (sort_type == REVERSE_SORTED)
    {
        for (i = 0; i < length; i++)
        {
            arr[i] = length - i - 1;
        }
    }
}

bool is_sorted(int *arr, int arr_size)
{
    for (int i = 1; i < arr_size; i++)
    {
        if (arr[i] < arr[i - 1])
        {
            return false;
        }
    }

    return true;
}

__device__ solve(int **tempList, int left_start, int right_start, int old_left_start, int left_end, int right_end, int headLoc)
{
    for (int i = 0; i < walkLen; i++)
    {
        if (tempList[current_list][left_start] < tempList[current_list][right_start])
        {
            tempList[!current_list][headLoc] = tempList[current_list][left_start];
            left_start++;
            headLoc++;

            // Check if l is now empty
            if (left_start == left_end)
            {
                // place the left over elements into the array
                for (int j = right_start; j < right_end; j++)
                {
                    tempList[!current_list][headLoc] = tempList[current_list][right_start];
                    right_start++;
                    headLoc++;
                }
            }
        }
        else
        {
            tempList[!current_list][headLoc] = tempList[current_list][right_start];
            right_start++;
            // Check if r is now empty
            if (right_start == right_end)
            {
                // place the left over elements into the array
                for (int j = left_start; j < left_end; j++)
                {
                    tempList[!current_list][headLoc] = tempList[current_list][right_start];
                    right_start++;
                    headLoc++;
                }
            }
        }
    }
}

__global__ void device_merge(int *device_vals, int length, int elements_per_thread)
{

    int thread_start, thread_end;

    int left_start, right_start; // left and right lists will be merged
    int old_left_start;
    int left_end, right_end;
    int headLoc = 0;      // current location of the write head on the newList
    int current_list = 0; // two lists, one is current

    // allocate enough shared memory for this block's list...

    // __shared__ float tempList[2][SHARED / sizeof(float)]; // FIX, maybe j length * sizeof(int)?
    __shared__ int tempList[2][block * sizeof(int)]; // maintains two parallel lists, one used to store merged results

    // Load memory
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < elements_per_thread; i++)
    {
        if (index + i < length)
        {
            tempList[current_list][elements_per_thread * threadIdx.x + i] = device_vals[index + i];
        }
    }

    // Wait until all memory has been loaded
    __syncthreads();

    // Merge the left and right lists.
    for (int walkLen = 1; walkLen < length; walkLen *= 2)
    {
        // Set up start and end indices.
        thread_start = elements_per_thread * threadIdx.x;
        thread_end = thread_start + elements_per_thread;
        left_start = thread_start;

        while (left_start < thread_end)
        {
            old_left_start = left_start;
            // If this happens, we are done.
            // if (left_start > thread_end)
            // {
            //     left_start = length;
            //     break;
            // }

            left_end = left_start + walkLen;
            if (left_end > thread_end)
            {
                left_end = length;
            }

            right_start = left_end;
            if (right_start > thread_end)
            {
                right_end = length;
            }

            right_end = right_start + walkLen;
            if (right_end > thread_end)
            {
                right_end = length;
            }

            solve(&tempList, left_start, right_start, old_left_start, left_end, right_end, headLoc);
            left_start = old_left_start + 2 * walkLen;
            current_list = !current_list;
        }
    }
    // Wait until all thread completes swapping if not race condition will appear
    // as it might update non sorted value to device_vals
    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < elements_per_thread; i++)
    {
        if (index + i < length)
        {
            device_vals[index + i] = subList[current_list][elements_per_thread * threadIdx.x + i];
        }
    }
    // Wait until all memory has been loaded
    __syncthreads();

    return;
}

void mergesort(int *host_list, int len, int threads, int blocks)
{

    // device copy
    int *device_list;
    // Allocate space for device copy
    cudaMalloc((void **)&device_list, len * sizeof(int));
    // copy input to device
    cudaMemcpy(device_list, host_list, len * sizeof(int), cudaMemcpyHostToDevice);

    // int elementsPerThread = ceil(len / float(threads * blocks));
    int elements_per_thread = 1;

    // Launch a Device_Merge kernel on GPU
    Device_Merge<<<blocks, threads>>>(device_list, len, elements_per_thread);

    cudaMemcpy(host_list, device_list, len * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(device_list);
}

int main(int argc, char *argv[])
{
    NUM_VALS = atoi(argv[1]);
    THREADS = atoi(argv[2]);
    BLOCKS = NUM_VALS / THREADS;

    printf("Number of threads: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);

    int *values = (int *)malloc(NUM_VALS * sizeof(int));
    array_fill(values, NUM_VALS, RANDOM);

    printArray(values, NUM_VALS);

    mergesort(values, NUM_VALS, THREADS, BLOCKS);

    printArray(values, NUM_VALS);
    cout << "Sorted?: " << is_sorted(values, NUM_VALS) << endl;
}