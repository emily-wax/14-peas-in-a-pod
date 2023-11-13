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

int THREADS;
int BLOCKS;
int NUM_VALS;

const char *bitonic_sort_step_region = "bitonic_sort_step";
const char *cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char *cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";

int kernel_calls = 0;
float effective_bandwidth_gb_s;
float bitonic_sort_step_time;
float cudaMemcpy_host_to_device_time;
float cudaMemcpy_device_to_host_time;

void print_elapsed(clock_t start, clock_t stop)
{
    double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.3fs\n", elapsed);
}

float random_float()
{
    return (float)rand() / (float)RAND_MAX;
}

void array_print(float *arr, int length)
{
    int i;
    for (i = 0; i < length; ++i)
    {
        printf("%1.3f ", arr[i]);
    }
    printf("\n");
}

void array_fill(float *arr, int length)
{
    srand(time(NULL));
    int i;
    for (i = 0; i < length; ++i)
    {
        arr[i] = random_float();
    }
}

__global__ void bitonic_sort_step(float *dev_values, int j, int k)
{
    unsigned int i, ixj; /* Sorting partners: i and ixj */
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i ^ j;

    /* The threads with the lowest ids sort the array. */
    if ((ixj) > i)
    {
        if ((i & k) == 0)
        {
            /* Sort ascending */
            if (dev_values[i] > dev_values[ixj])
            {
                /* exchange(i,ixj); */
                float temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
        if ((i & k) != 0)
        {
            /* Sort descending */
            if (dev_values[i] < dev_values[ixj])
            {
                /* exchange(i,ixj); */
                float temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
    }
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(float *values)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *dev_values;
    size_t size = NUM_VALS * sizeof(float);

    cudaMalloc((void **)&dev_values, size);

    // MEM COPY FROM HOST TO DEVICE
    CALI_MARK_BEGIN(cudaMemcpy_host_to_device);
    cudaEventRecord(start);
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cudaMemcpy_host_to_device_time, start, stop);
    CALI_MARK_END(cudaMemcpy_host_to_device);

    dim3 blocks(BLOCKS, 1);   /* Number of blocks   */
    dim3 threads(THREADS, 1); /* Number of threads  */

    int j, k;
    CALI_MARK_BEGIN(bitonic_sort_step_region);
    cudaEventRecord(start);
    /* Major step */
    for (k = 2; k <= NUM_VALS; k <<= 1)
    {
        /* Minor step */
        for (j = k >> 1; j > 0; j = j >> 1)
        {
            kernel_calls += 1;
            bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&bitonic_sort_step_time, start, stop);
    CALI_MARK_END(bitonic_sort_step_region);

    // MEM COPY FROM DEVICE TO HOST
    CALI_MARK_BEGIN(cudaMemcpy_device_to_host);
    cudaEventRecord(start);
    cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cudaMemcpy_device_to_host_time, start, stop);
    CALI_MARK_END(cudaMemcpy_device_to_host);

    cudaFree(dev_values);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char *argv[])
{
    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    BLOCKS = NUM_VALS / THREADS;

    printf("Number of threads: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    clock_t start, stop;

    float *values = (float *)malloc(NUM_VALS * sizeof(float));
    array_fill(values, NUM_VALS);

    start = clock();
    bitonic_sort(values); /* Inplace */
    stop = clock();

    print_elapsed(start, stop);

    effective_bandwidth_gb_s = (kernel_calls * 6 * NUM_VALS * sizeof(float) / 1000000000) / (bitonic_sort_step_time / 1000);
    printf("CUDA host to device time (ms): %f\n", cudaMemcpy_host_to_device_time);
    printf("CUDA bitonic sort step time (ms): %f\n", bitonic_sort_step_time);
    printf("CUDA device to host time (ms): %f\n", cudaMemcpy_device_to_host_time);
    printf("Kernel calls: %d\n", kernel_calls);
    printf("Effective bandwith (GB/s): %f\n", effective_bandwidth_gb_s);

    // Store results in these variables.
    /*
    float effective_bandwidth_gb_s;
    float bitonic_sort_step_time;
    float cudaMemcpy_host_to_device_time;
    float cudaMemcpy_device_to_host_time;
    */

    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("num_threads", THREADS);
    adiak::value("num_blocks", BLOCKS);
    adiak::value("num_vals", NUM_VALS);
    adiak::value("program_name", "cuda_bitonic_sort");
    adiak::value("datatype_size", sizeof(float));
    adiak::value("effective_bandwidth (GB/s)", effective_bandwidth_gb_s);
    adiak::value("bitonic_sort_step_time", bitonic_sort_step_time);
    adiak::value("cudaMemcpy_host_to_device_time", cudaMemcpy_host_to_device_time);
    adiak::value("cudaMemcpy_device_to_host_time", cudaMemcpy_device_to_host_time);

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();
}