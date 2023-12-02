/******************************************************************************
 * FILE: mergesort.cpp
 * DESCRIPTION:
 *   Parallelized merge sort algorithm using CUDA
 * AUTHOR: Harini Kumar
 * LAST REVISED: 12/1/23
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
    int start = width * idx;                         // used to index into array at start of slice
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

    dim3 blocks(BLOCKS, 1);   // number of blocks
    dim3 threads(THREADS, 1); // number of threads

    // allocate two arrays that are size of og input array and copy input into one
    cudaMalloc((void **)&d_data, size * sizeof(int));
    cudaMalloc((void **)&d_swp, size * sizeof(int));

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    cudaMemcpy(d_data, data, size * sizeof(int), cudaMemcpyHostToDevice);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    // used to point to two arrays to make swapping easier
    int *A = d_data;
    int *B = d_swp;

    // similar to mpi merge, slices start small and grow in size as you progress through merge tree
    for (int width = 2; width < (size << 1); width <<= 1) // slice width multiplied by 2 each time, ends when width is og input size
    {
        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");
        gpu_mergesort<<<blocks, threads>>>(A, B, size, width);
        CALI_MARK_END("comp_large");
        CALI_MARK_END("comp");

        // swap A and B each time following gpu_mergesort to correctly track which is most updated merged
        A = A == d_data ? d_swp : d_data;
        B = B == d_data ? d_swp : d_data;
    }

    cudaDeviceSynchronize();

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    // merged list copied back to host
    cudaMemcpy(data, A, size * sizeof(int), cudaMemcpyDeviceToHost);
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    cudaFree(A);
    cudaFree(B);
}

int main(int argc, char *argv[])
{
    CALI_MARK_BEGIN("main");

    NUM_VALS = atoi(argv[1]);
    THREADS = atoi(argv[2]);
    BLOCKS = NUM_VALS / THREADS;
    int input_type = atoi(argv[3]);

    printf("Number of threads: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);

    CALI_MARK_BEGIN("data_init");
    int *values = (int *)malloc(NUM_VALS * sizeof(int));
    fillArray(values, NUM_VALS, input_type);
    CALI_MARK_END("data_init");

    // printArray(values, NUM_VALS);

    mergesort(values, NUM_VALS);

    // printArray(values, NUM_VALS);
    CALI_MARK_BEGIN("correctness_check");
    cout << "Sorted?: " << is_sorted(values, NUM_VALS) << endl;
    CALI_MARK_END("correctness_check");

    const char *algorithm = "MergeSort";
    const char *programmingModel = "CUDA";
    const char *datatype = "int";
    int sizeOfDatatype = sizeof(int);
    int inputSize = NUM_VALS;
    // const char *inputType = "Random";
    static const char *const inputTypes[] = {"Sorted", "ReverseSorted", "1%perturbed", "Random"};
    const char *num_procs = "N/A";
    int num_threads = THREADS;
    int num_blocks = BLOCKS;
    int group_number = 14;
    const char *implementation_source = "Online";

    adiak::init(NULL);
    adiak::launchdate();                                          // launch date of the job
    adiak::libraries();                                           // Libraries used
    adiak::cmdline();                                             // Command line used to launch the job
    adiak::clustername();                                         // Name of the cluster
    adiak::value("Algorithm", algorithm);                         // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", programmingModel);           // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", datatype);                           // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeOfDatatype);               // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", inputSize);                         // The number of elements in input dataset (1000)
    adiak::value("InputType", inputTypes[input_type]);            // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_procs);                         // The number of processors (MPI ranks)
    adiak::value("num_threads", num_threads);                     // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", num_blocks);                       // The number of CUDA blocks
    adiak::value("group_num", group_number);                      // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", implementation_source); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    CALI_MARK_END("main");
}