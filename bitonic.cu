/*
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 * License: BSD 3
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int THREADS;
int BLOCKS;
int NUM_VALS;

const char* bitonic_sort_step_region = "bitonic_sort_step";
const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";

cudaEvent_t bitonic_sort_step_start_time;
cudaEvent_t bitonic_sort_step_end_time;
cudaEvent_t host_to_device_start_time;
cudaEvent_t host_to_device_end_time;
cudaEvent_t device_to_host_start_time;
cudaEvent_t device_to_host_end_time;

enum sort_type{
  SORTED,
  REVERSE_SORTED,
  PERTURBED,
  RANDOM
};


bool check_array(float* arr, int length){
  for (int i = 0; i < length -1; i++){
    if (arr[i] > arr[i+1]){
      return false;
    }
  }
  return true; 
}

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

void array_print(float *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
}

void array_fill(float *arr, int length, int sort_type)
{
  srand(time(NULL));
  int i;
  if (sort_type == RANDOM){
    for (i = 0; i < length; ++i) {
      arr[i] = random_float();
    } 
  }
  else if (sort_type == SORTED){
    for (i = 0; i < length; i++){
      arr[i] = i;
    }
  }
  else if (sort_type == PERTURBED){
    for(i = 0; i < length; i++){
      arr[i] = i;
      int temp = rand() % 100;
      if (temp == 1){
        arr[i] = rand() % length; 
      }
    }
  }
  else if (sort_type == REVERSE_SORTED){
    for (i = 0; i < length; i++){
      arr[i] = length - i - 1;
    }
  }
}

__global__ void bitonic_sort_step(float *dev_values, int j, int k)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
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
int bitonic_sort(float *values)
{
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);

  cudaMalloc((void**) &dev_values, size);
  
  //MEM COPY FROM HOST TO DEVICE
  CALI_MARK_BEGIN(cudaMemcpy_host_to_device);
  cudaEventRecord(host_to_device_start_time);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  cudaEventRecord(host_to_device_end_time);
  CALI_MARK_END(cudaMemcpy_host_to_device);
  cudaEventSynchronize(host_to_device_end_time);

  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */
  
  int j, k;
  int count = 0;
  /* Major step */
  CALI_MARK_BEGIN(bitonic_sort_step_region);
  cudaEventRecord(bitonic_sort_step_start_time);
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      count++;
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
    }
  }
  cudaDeviceSynchronize();
  cudaEventRecord(bitonic_sort_step_end_time);
  CALI_MARK_END(bitonic_sort_step_region);
  cudaEventSynchronize(bitonic_sort_step_end_time);
  
  //MEM COPY FROM DEVICE TO HOST
  CALI_MARK_BEGIN(cudaMemcpy_device_to_host);
  cudaEventRecord(device_to_host_start_time);
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaEventRecord(device_to_host_end_time);
  CALI_MARK_END(cudaMemcpy_device_to_host);
  cudaFree(dev_values);
  cudaEventSynchronize(device_to_host_end_time);

  printf("Count is: %d \n", count);
  return count;
}

int main(int argc, char *argv[])
{
  cudaEventCreate(&bitonic_sort_step_start_time);
  cudaEventCreate(&bitonic_sort_step_end_time);
  cudaEventCreate(&host_to_device_start_time);
  cudaEventCreate(&host_to_device_end_time);
  cudaEventCreate(&device_to_host_start_time);
  cudaEventCreate(&device_to_host_end_time);

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
  float c = 0;

  float *values = (float*) malloc( NUM_VALS * sizeof(float));
  array_fill(values, NUM_VALS, RANDOM);

  array_print(values, NUM_VALS); 

  start = clock();
  c = bitonic_sort(values); /* Inplace */
  stop = clock();

  print_elapsed(start, stop);

  array_print(values, NUM_VALS);
  if (!check_array(values, NUM_VALS)){
    printf("ERROR ARRAY IS NOT SORTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n\n\n");
  }

  // Store results in these variables.
  float effective_bandwidth_gb_s;
  float bitonic_sort_step_time;
  float cudaMemcpy_host_to_device_time;
  float cudaMemcpy_device_to_host_time;


  cudaEventElapsedTime(&bitonic_sort_step_time, bitonic_sort_step_start_time, bitonic_sort_step_end_time);
  cudaEventElapsedTime(&cudaMemcpy_host_to_device_time, host_to_device_start_time, host_to_device_end_time);
  cudaEventElapsedTime(&cudaMemcpy_device_to_host_time, device_to_host_start_time, device_to_host_end_time);

  bitonic_sort_step_time /= 1000;
  cudaMemcpy_device_to_host_time /= 1000;
  cudaMemcpy_host_to_device_time /= 1000;
  float temp = (c*2*4*NUM_VALS) /(bitonic_sort_step_time);
  effective_bandwidth_gb_s = temp/1e9;

  printf("bitonic sort step time: %f \n", bitonic_sort_step_time);
  printf("host to device time: %f \n", cudaMemcpy_host_to_device_time);
  printf("device to host time: %f \n", cudaMemcpy_device_to_host_time);
  printf("effective bandwith time: %f \n", effective_bandwidth_gb_s);
  printf("Count is: %d \n", c);


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