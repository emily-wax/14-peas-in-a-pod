/******************************************************************************
 * FILE: mergesort.cpp
 * DESCRIPTION:
 *   Parallelized merge sort algorithm using CUDA
 * AUTHOR: Harini Kumar
 * LAST REVISED: 11/15/23
 ******************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda_runtime.h>
#include <cuda.h>
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

void fillArray(int *arr, int length, int sort_type)
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

// called for each slice, merges source left array starting at start and right array starting at middle
// results stored in dest, slice indices maintained
__device__ void gpu_merge(int *source, int *dest, int start, int middle, int end)
{
    int left_ptr = start;
    int right_ptr = middle;
    for (int merge_ptr = start; merge_ptr < end; merge_ptr++)
    {
        if (left_ptr < middle && (right_ptr >= end || source[left_ptr] < source[right_ptr]))
        {
            dest[merge_ptr] = source[left_ptr];
            left_ptr++;
        }
        else
        {
            dest[merge_ptr] = source[right_ptr];
            right_ptr++;
        }
    }
}

// mergesort for the slice given to device
__global__ void gpu_mergesort(int *source, int *dest, int size, int width)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x; // calculate unique index across threads and blocks
    int start = width * idx;                         // used to index into array
    int middle;
    int end;

    if (start >= size)
        return;

    middle = min(start + (width >> 1), size);
    end = min(start + width, size);
    gpu_merge(source, dest, start, middle, end);
}

// called by host/main
void mergesort(int *data, int size)
{

    // will switch btwn following two arrays when merging (one holds most updated merged, one holds not)
    int *d_data;
    int *d_swp;

    dim3 *d_threads;
    dim3 *d_blocks;

    dim3 blocks(BLOCKS, 1);   // number of blocks
    dim3 threads(THREADS, 1); // number of threads

    // allocate two arrays that are size of og input array and copy input into one
    cudaMalloc((void **)&d_data, size * sizeof(int));
    cudaMalloc((void **)&d_swp, size * sizeof(int));

    cudaMemcpy(d_data, data, size * sizeof(int), cudaMemcpyHostToDevice);

    // used to point to two arrays to make swapping easier
    int *A = d_data;
    int *B = d_swp;

    // int nThreads = NUM_VALS; // threads * blocks

    // similar to mpi merge, slices start small and grow in size as you progress through merge tree
    for (int width = 2; width < (size * 2); width *= 2) // slice width multiplied by 2 each time, ends when width is og input size
    {
        gpu_mergesort<<<blocks, threads>>>(A, B, size, width);

        // swap A and B each time following gpu_mergesort to correctly track which is most updated merged
        A = A == d_data ? d_swp : d_data;
        B = B == d_data ? d_swp : d_data;
    }

    // merged list copied back to host
    cudaMemcpy(data, A, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(A);
    cudaFree(B);
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
    fillArray(values, NUM_VALS, RANDOM);

    // printArray(values, NUM_VALS);

    mergesort(values, NUM_VALS);

    // printArray(values, NUM_VALS);
    cout << "Sorted?: " << is_sorted(values, NUM_VALS) << endl;
}
