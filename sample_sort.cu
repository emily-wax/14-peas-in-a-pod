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

void array_print(int *arr, int length) 
{
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

  /* find index using block size and thread id */
  for(int i = 0; i < (num_threads - 1); i++){
    curr_splitter = data[(num_vals/(num_threads * num_threads) * (i + 1)) + start_index ];
    splitters[ ( (num_threads - 1) * thread_id) + i ] = curr_splitter;
  }


  /* pick splitters for block and put in splitters array */

}

int main(int argc, char *argv[]){

  int * dev_values;
  int * dev_splitters;

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

  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */

  array_fill(values, NUM_VALS, REVERSE_SORTED);

  array_print(values, NUM_VALS);

  /* Cuda mallocs for values and all splitters arrays */
  cudaMalloc((void**) &dev_values, NUM_VALS * sizeof(int));
  cudaMalloc((void**) &dev_splitters, sizeof(int) * (THREADS - 1) * THREADS );

  /* Memcpy from host to device */
  cudaMemcpy(dev_values, values, NUM_VALS * sizeof(int), cudaMemcpyHostToDevice);
  // EW TODO: do I have to memcpy for splitters when there isn't anything in there?

  /* <<<numBlocks, threadsPerBlock>>> */
  localSplitters<<<blocks, threads>>>(dev_values, dev_splitters, THREADS, NUM_VALS, BLOCKS);

  /* Memcpy from devie to host */
  cudaMemcpy(all_splitters, dev_splitters, sizeof(int) * (THREADS - 1) * THREADS, cudaMemcpyDeviceToHost);

  /* sort all splitters and choose global */
  qsort((char *) all_splitters, THREADS * (THREADS - 1), sizeof(int), intCompare);

  for(int i = 0; i < THREADS - 1; i++)
  {
    global_splitters[i] = all_splitters[(THREADS - 1) * (i + 1)];
  }

  array_print(all_splitters, (THREADS - 1) * THREADS);
  array_print(global_splitters, THREADS - 1);

  // have a function that calculates the offsets (determines starting point for each bucket)

  // run prefix sum to calculate starting point

  // distribute data (use atomic add with size of buckets until offset and size match)

  // use the offsets and size to accurately place into sorted data


  /* free device memory */
  cudaFree(dev_splitters);
  cudaFree(dev_values);

  if (!check_array(values, NUM_VALS)){
    printf("ERROR ARRAY IS NOT SORTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n\n\n");
  }

}