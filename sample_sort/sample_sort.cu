 /******************************************************************************
* FILE: sample_sort.cu
* DESCRIPTION:  
*   Sample sort CUDA implementation
* AUTHOR: Emily Wax
* LAST REVISED: 11/12/23
******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

int THREADS;
int BLOCKS;
int NUM_VALS;

/* Sort Types */
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

/* used after sorting is complete, checks if array has 
 * been sucessfully sorted */
bool check_array(int* arr, int length){
  for (int i = 0; i < length -1; i++){
    if (arr[i] > arr[i+1]){
      return false;
    }
  }
  return true; 
}

void array_print(int *arr, int length) {
  int i;
  printf( "print array: \n");
  for (i = 0; i < length; ++i) {
    printf("%d ",  arr[i]);
  }
  printf("\n");
}

/* fills array with values depending on desired sort type*/
void array_fill(int *arr, int length, int sort_type){
  srand(time(NULL));
  int i;
  if (sort_type == RANDOM){
    for (i = 0; i < length; ++i) {
      arr[i] = rand() % (INT_MAX);
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


__global__ void localSplitters( const int* data, int* splitters, int num_threads, int num_vals, int block_size ){

  unsigned int thread_id;
  int start_index;
  int curr_splitter;

  thread_id = threadIdx.x + blockDim.x * blockIdx.x;

  // start and end index are inclusive
  start_index = thread_id * block_size;

  /* pick splitters for block and put in splitters array */
  for(int i = 0; i < (num_threads - 1); i++){
    curr_splitter = data[(num_vals/(num_threads * num_threads) * (i + 1)) + start_index ];
    splitters[ ( (num_threads - 1) * thread_id) + i ] = curr_splitter;
  }

}

__global__ void getBucketSize( const int* data, int* splitters, int* bucket_caps, int num_threads, int block_size ){

  unsigned int thread_id;
  int start_index;
  int value;

  thread_id = threadIdx.x + blockDim.x * blockIdx.x;

  // get start index of block (inclusive)
  start_index = thread_id * block_size;

  // decide which bucket it goes in (outer loop through block, inner loop through splitters)
  for( int j = 0; j < block_size; j++)
  {
    value = data[start_index + j ];
    for(int i = 0; i < num_threads - 1; i++)
    {
      if( value < splitters[ i ] )
      {
        atomicAdd(&bucket_caps[i], 1);
        break;
      }

      // atomically increase that bucket's size
      if( i == num_threads - 2 )
      {
        atomicAdd(&bucket_caps[i + 1], 1);
      }
    }
  }
}

__global__ void distributeData( const int* data, int* sorted_array, int* splitters, int* bucket_starts, int* bucket_sizes, int num_threads, int block_size){

  unsigned int thread_id;
  int start_index;
  int value;

  thread_id = threadIdx.x + blockDim.x * blockIdx.x;

  // get start index of block (inclusive)
  start_index = thread_id * block_size;

  // see which bucket element belongs in
  for( int j = 0; j < block_size; j++)
  {
    value = data[start_index + j ];
    for(int i = 0; i < num_threads - 1; i++)
    {
      if( value < splitters[ i ] )
      {
        sorted_array[bucket_starts[i] + bucket_sizes[i]] = value;
        atomicAdd( &bucket_sizes[i], 1);
        break;
      }

      // atomically increase that bucket's size
      if( i == num_threads - 2 )
      {
        sorted_array[bucket_starts[i + 1] + bucket_sizes[i + 1]] = value;
        atomicAdd(&bucket_sizes[i + 1], 1);
      }
    }
  }

  __syncthreads();

}

int main(int argc, char *argv[]){

  int * dev_values;
  int * dev_splitters;
  int * dev_bucket_caps;
  int * dev_global_splitters;
  int * dev_sorted;
  int * dev_starts;
  int * dev_sizes;
  int prefix_sum = 0;

  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  BLOCKS = NUM_VALS / THREADS;

  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Size of blocks: %d\n", BLOCKS);

  // allocate space for values and splitters
  int *values = (int*) malloc( NUM_VALS * sizeof(int));
  int *all_splitters = (int *)malloc( sizeof(int) * (THREADS - 1) * THREADS);
  int *global_splitters = (int *)malloc( sizeof(int) * (THREADS - 1));
  int *bucket_caps = (int *)malloc(sizeof(int) * (THREADS));
  int *bucket_starts = (int *)malloc(sizeof(int) * (THREADS));
  int *bucket_sizes = (int*)malloc(sizeof(int) * THREADS);
  int *sorted_array = (int*) malloc( NUM_VALS * sizeof(int));

  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */

  array_fill(values, NUM_VALS, SORTED);

  array_print(values, NUM_VALS);

  /* Cuda mallocs for values and all splitters arrays */
  cudaMalloc((void**) &dev_values, NUM_VALS * sizeof(int));
  cudaMalloc((void**) &dev_splitters, sizeof(int) * (THREADS - 1) * THREADS );

  /* Memcpy from host to device */
  cudaMemcpy(dev_values, values, NUM_VALS * sizeof(int), cudaMemcpyHostToDevice);
  // EW TODO: do I have to memcpy for splitters when there isn't anything in there?

  /* <<<numBlocks, threadsPerBlock>>> */
  localSplitters<<<1, threads>>>(dev_values, dev_splitters, THREADS, NUM_VALS, BLOCKS);

  /* Memcpy from devie to host */
  cudaMemcpy(all_splitters, dev_splitters, sizeof(int) * (THREADS - 1) * THREADS, cudaMemcpyDeviceToHost);

  cudaFree(dev_splitters);

  /* sort all splitters and choose global */
  qsort((char *) all_splitters, THREADS * (THREADS - 1), sizeof(int), intCompare);

  for(int i = 0; i < THREADS - 1; i++)
  {
    global_splitters[i] = all_splitters[(THREADS - 1) * (i + 1)];
  }

  array_print(all_splitters, (THREADS - 1) * THREADS);
  array_print(global_splitters, THREADS - 1);

  // have a function that calculates the offsets (determines starting point for each bucket)
  cudaMalloc((void**) &dev_bucket_caps, THREADS * sizeof(int));
  cudaMalloc((void**) &dev_global_splitters, sizeof(int) * (THREADS - 1) );

  /* host to device*/
  cudaMemcpy( dev_bucket_caps, bucket_caps, sizeof(int) * THREADS, cudaMemcpyHostToDevice);
  cudaMemcpy( dev_global_splitters, global_splitters, sizeof(int) * (THREADS - 1), cudaMemcpyHostToDevice);

  getBucketSize<<<1, threads>>>(dev_values, dev_global_splitters, dev_bucket_caps, THREADS, BLOCKS);

  cudaMemcpy(bucket_caps, dev_bucket_caps, THREADS * sizeof(int), cudaMemcpyDeviceToHost);

  array_print(bucket_caps, THREADS);

  // run prefix sum to calculate starting point
  bucket_starts[0] = 0;

  for(int i = 1; i < THREADS; i++){
    prefix_sum += bucket_caps[i - 1];
    bucket_starts[i] = prefix_sum;
  }

  array_print(bucket_starts, THREADS);

  // distribute data (use atomic add with size of buckets until offset and size match)
  cudaMalloc((void**) &dev_sorted, NUM_VALS * sizeof(int));
  cudaMalloc((void**) &dev_starts, THREADS * sizeof(int));
  cudaMalloc((void**) &dev_sizes, THREADS* sizeof(int));

  // use the offsets and size to accurately place into sorted data

  cudaMemcpy(dev_starts, bucket_starts, THREADS * sizeof(int), cudaMemcpyHostToDevice);

  distributeData<<<1, threads>>>(dev_values, dev_sorted, dev_global_splitters, dev_starts, dev_sizes, THREADS, BLOCKS);

  cudaMemcpy( sorted_array, dev_sorted, NUM_VALS * sizeof(int), cudaMemcpyDeviceToHost);

  // print sorted array
  array_print(sorted_array, NUM_VALS);

  /* free device memory */
  cudaFree(dev_values);
  cudaFree(dev_global_splitters);
  cudaFree(dev_bucket_caps);


  if (!check_array(sorted_array, NUM_VALS)){
    printf("ERROR ARRAY IS NOT SORTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n\n\n");
  }

}