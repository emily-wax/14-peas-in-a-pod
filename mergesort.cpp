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

using namespace std;

enum sort_type
{
    SORTED,
    REVERSE_SORTED,
    PERTURBED,
    RANDOM
};

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

void merge(int *merged, int *left_arr, int *right_arr, int left_size, int right_size)
{
    int left_ptr = 0;
    int right_ptr = 0;
    int merged_ptr = left_ptr;

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

void sequential_merge()
{
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
}