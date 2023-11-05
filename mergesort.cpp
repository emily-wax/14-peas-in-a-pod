/******************************************************************************
* FILE: mergesort.cpp
* DESCRIPTION:  
*   Parallelized merge sort algorithm using MPI
* AUTHOR: Harini Kumar
* LAST REVISED: 11/02/23
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <iostream>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <bits/stdc++.h>

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

void fillArray(int *values, int block_size, int NUM_VALS, int sort_type)
{ // work each thread will do
    int thread_id, num_threads;
    MPI_Comm_rank(MPI_COMM_WORLD, &thread_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_threads);

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
        int i = start_index;
        while (end_val >= start_val)
        {
            values[i] = end_val;
            end_val--;
            i++;
        }
    }
    else if (sort_type == RANDOM)
    {
        int i = start_index;
        while (i <= end_index)
        {
            values[i] = rand() % (INT_MAX);
            // values[i] = rand() % (100);
            i++;
        }
    }
    else if (sort_type == PERTURBED)
    {
        int i = start_index;
        int perturb_check;
        while (i <= end_index)
        {
            values[i] = start_val;
            perturb_check = rand() % 100;
            if (perturb_check == 1)
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
    int thread_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &thread_id);
    int block_size = NUM_VALS / numThreads;

    // moved this to main
    // if (thread_id == 0)
    // {
    //     values_array = (int *)malloc(NUM_VALS * sizeof(int));
    // }

    int *thread_values_array = (int *)malloc(block_size * sizeof(int));

    fillArray(thread_values_array, block_size, NUM_VALS, sortType);

    MPI_Gather(thread_values_array, block_size, MPI_INT, values_array, block_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (thread_id == 0)
    {
        printArray(values_array, NUM_VALS);
    }
}

void merge_inplace(int *merged, int *left_arr, int *right_arr, int left_size, int right_size, int arr_ptr)
{
    int left_ptr = 0;
    int right_ptr = 0;
    int merged_ptr = arr_ptr;

    while (left_ptr < left_size && right_ptr < right_size)
    {
        if (left_arr[left_ptr] <= right_arr[right_ptr])
        {
            merged[merged_ptr] = left_arr[left_ptr];
            left_ptr++;
        }
        else
        {
            merged[merged_ptr] = right_arr[right_ptr];
            right_ptr++;
        }
        merged_ptr++;
    }

    while (left_ptr < left_size)
    {
        merged[merged_ptr] = left_arr[left_ptr];
        left_ptr += 1;
        merged_ptr += 1;
    }
    while (right_ptr < right_size)
    {
        merged[merged_ptr] = right_arr[right_ptr];
        right_ptr += 1;
        merged_ptr += 1;
    }
}

void sequential_mergesort(int *arr, int left, int right)
{
    if (left >= right)
    {
        return;
    }

    int mid = left + (right - left) / 2;

    sequential_mergesort(arr, left, mid);
    sequential_mergesort(arr, mid + 1, right);

    int left_arr[mid - left + 1];
    int right_arr[right - mid];

    for (int i = 0; i < mid - left + 1; i++)
    {
        left_arr[i] = arr[left + i];
    }
    for (int i = 0; i < right - mid; i++)
    {
        right_arr[i] = arr[mid + 1 + i];
    }

    merge_inplace(arr, left_arr, right_arr, mid - left + 1, right - mid, left);
}

int *merge(int *left_arr, int *right_arr, int left_size, int right_size)
{
    int left_ptr = 0;
    int right_ptr = 0;
    int merged_ptr = 0;

    int *merged = (int *)malloc((left_size + right_size) * sizeof(int));

    while (left_ptr < left_size && right_ptr < right_size)
    {
        if (left_arr[left_ptr] <= right_arr[right_ptr])
        {
            merged[merged_ptr] = left_arr[left_ptr];
            left_ptr++;
        }
        else
        {
            merged[merged_ptr] = right_arr[right_ptr];
            right_ptr++;
        }
        merged_ptr++;
    }

    while (left_ptr < left_size)
    {
        merged[merged_ptr] = left_arr[left_ptr];
        left_ptr += 1;
        merged_ptr += 1;
    }
    while (right_ptr < right_size)
    {
        merged[merged_ptr] = right_arr[right_ptr];
        right_ptr += 1;
        merged_ptr += 1;
    }

    return merged;
}

void mergesort(int tree_height, int thread_id, int *thread_array, int arr_size, MPI_Comm comm, int **global_array)
{
    int curr_height = 0;
    int *left_data = thread_array;
    int *right_data = nullptr;
    int *merged_data = nullptr;
    int block_size = arr_size;

    while (curr_height < tree_height)
    {
        // check if left or right "parent" in merge tree
        bool is_left_branch = thread_id % (1 << (curr_height + 1)) == 0; // thread_id % 2^(curr_height + 1) == 0

        if (is_left_branch)
        {
            // find corresponding right branch id
            int right_branch = thread_id + (1 << curr_height); // thread_id + 2^curr_height

            // receive right branch's data
            right_data = (int *)malloc(arr_size * sizeof(int));
            MPI_Recv(right_data, arr_size, MPI_INT, right_branch, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // merge two branches' data
            merged_data = (int *)malloc(2 * arr_size * sizeof(int));
            merged_data = merge(left_data, right_data, arr_size, arr_size);

            // update info for future while loop iterations
            left_data = merged_data; // since left branch is the one that will continue working
            arr_size *= 2;
            delete[] right_data;
            merged_data = nullptr;
            curr_height += 1;
        }
        else
        { // right branch
            // find corresponding left branch id and send data to it
            int left_branch = thread_id - (1 << curr_height); // thread_id - 2^curr_height
            MPI_Send(left_data, arr_size, MPI_INT, left_branch, 0, MPI_COMM_WORLD);
            delete[] left_data;        // holding data that has been sent to left branch, not needed
            curr_height = tree_height; // while loop terminates for right branches, number of active threads halves at each level of tree
        }
    }

    if (thread_id == 0)
    {
        *global_array = left_data;
    }
}

int main(int argc, char **argv)
{
    int NUM_VALS = atoi(argv[1]);
    int num_threads;

    int thread_id;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_threads);
    MPI_Comm_rank(MPI_COMM_WORLD, &thread_id);

    int block_size = NUM_VALS / num_threads;

    int *values_array_global = nullptr;
    if (thread_id == 0)
    {
        values_array_global = (int *)malloc(NUM_VALS * sizeof(int));
    }

    // should be done by all threads not just root, only yields one gathered global array
    createData(num_threads, values_array_global, NUM_VALS, RANDOM);
    MPI_Barrier(MPI_COMM_WORLD);

    // all threads given own block of array
    int *values_array_thread = (int *)malloc(block_size * sizeof(int));
    MPI_Scatter(values_array_global, block_size, MPI_INT, values_array_thread, block_size, MPI_INT, 0, MPI_COMM_WORLD);

    // sort each thread's array using sequential merge sort
    sequential_mergesort(values_array_thread, 0, block_size - 1);

    // call merge sort
    int merge_tree_height = log2(num_threads);
    mergesort(merge_tree_height, thread_id, values_array_thread, block_size, MPI_COMM_WORLD, &values_array_global);

    if (thread_id == 0)
    {
        printArray(values_array_global, NUM_VALS);
        delete[] values_array_global;
    }

    MPI_Finalize();
}