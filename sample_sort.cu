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
#include <string>


#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

int THREADS;
int BLOCKS;
int NUM_VALS;

/* Define Caliper Region Names*/
const char* main_cali = "main"; 
const char* data_init = "data_init";
const char* correctness_check = "correctness_check"; 
const char* comm = "comm"; 
const char* comm_small = "comm_small"; 
const char* comm_large = "comm_large"; 
const char* cudaMemcpy_region = "cudaMemcpy";
const char* comp = "comp";
const char* comp_small = "comp_small"; 
const char* comp_large = "comp_large"; 

// EW TODO: add cuda memcpy as a section ... check ansley's naming conventions

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
      arr[i] = rand() % (2000);
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

__global__ void distributeData( const int* data, int* sorted_array, int* splitters, int* bucket_starts, int* bucket_sizes, int num_threads, int block_size, int num_vals){

  unsigned int thread_id;
  int curr_index, end_index;
  int start_bound, end_bound;

  // plan:
  // iterate over full array, add to only your bucket by putting things in the sorted_array

  thread_id = threadIdx.x + blockDim.x * blockIdx.x;

  if(thread_id ==  num_threads - 1){
    end_index = num_vals;
  }else{
    end_index = bucket_starts[thread_id + 1];
  }

  // EW NOTE: bucket starts accounts for the bucket size calculated ... they are not the bounding values

  // right now, indexes are bucket bounds
  curr_index = bucket_starts[ thread_id ];

  // get start index of block (inclusive)
  if( thread_id == 0){
    start_bound = 0;
  } else{
    start_bound = splitters[thread_id - 1];
  }

  if( thread_id == (num_threads - 1)){
    // EW TODO: add num vals to parameters
    end_bound = INT_MAX;
  } else{
    end_bound = splitters[ thread_id ];
  }

  for(int i = 0; i < num_vals; i++){
    if( data[i] >= start_bound && data[i] < end_bound ){
      sorted_array[curr_index] = data[i];
      curr_index++;
    }
  }

  __syncthreads();

  // EW TODO: sort own bucket

    for (int i = bucket_starts[thread_id]; i < end_index; i++) {
        int key = sorted_array[i];
        int j = i - 1;
        while (j >= bucket_starts[thread_id] && sorted_array[j] > key) {
            sorted_array[j + 1] = sorted_array[j];
            j--;
        }
        sorted_array[j + 1] = key;
    }

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
  bool sorted;

  int sorttype = atoi(argv[3]);
  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);

  BLOCKS = NUM_VALS / THREADS;

  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Size of blocks: %d\n", BLOCKS);

  cali::ConfigManager mgr;
  mgr.start();

  CALI_MARK_BEGIN(main_cali);

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

  CALI_MARK_BEGIN(data_init);
  array_fill(values, NUM_VALS, sorttype);
  CALI_MARK_END(data_init);

  // array_print(values, NUM_VALS);

  /* Cuda mallocs for values and all splitters arrays */
  cudaMalloc((void**) &dev_values, NUM_VALS * sizeof(int));
  cudaMalloc((void**) &dev_splitters, sizeof(int) * (THREADS - 1) * THREADS );

  /* Memcpy from host to device */

  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(comm_large);
  CALI_MARK_BEGIN(cudaMemcpy_region);

  cudaMemcpy(dev_values, values, NUM_VALS * sizeof(int), cudaMemcpyHostToDevice);

  CALI_MARK_END(cudaMemcpy_region);
  CALI_MARK_END(comm_large);
  CALI_MARK_END(comm);
  // EW TODO: do I have to memcpy for splitters when there isn't anything in there?

  /* <<<numBlocks, threadsPerBlock>>> */
  CALI_MARK_BEGIN(comp);
  CALI_MARK_BEGIN(comp_small);
  localSplitters<<<blocks, threads>>>(dev_values, dev_splitters, THREADS, NUM_VALS, BLOCKS);
  CALI_MARK_END(comp_small);
  CALI_MARK_END(comp);

  /* Memcpy from device to host */
  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(comm_large);
  CALI_MARK_BEGIN(cudaMemcpy_region);

  cudaMemcpy(all_splitters, dev_splitters, sizeof(int) * (THREADS - 1) * THREADS, cudaMemcpyDeviceToHost);

  CALI_MARK_END(cudaMemcpy_region);
  CALI_MARK_END(comm_large);
  CALI_MARK_END(comm);

  cudaFree(dev_splitters);

  /* sort all splitters and choose global */
  CALI_MARK_BEGIN(comp);
  CALI_MARK_BEGIN(comp_small);
  qsort((char *) all_splitters, THREADS * (THREADS - 1), sizeof(int), intCompare);

  for(int i = 0; i < THREADS - 1; i++)
  {
    global_splitters[i] = all_splitters[(THREADS - 1) * (i + 1)];
  }
  CALI_MARK_END(comp_small);
  CALI_MARK_END(comp);

  // have a function that calculates the offsets (determines starting point for each bucket)
  cudaMalloc((void**) &dev_bucket_caps, THREADS * sizeof(int));
  cudaMalloc((void**) &dev_global_splitters, sizeof(int) * (THREADS - 1) );

  /* host to device*/

  for(int i = 0; i < THREADS; i++){
    bucket_caps[i] = 0;
  }

  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(comm_large);
  CALI_MARK_BEGIN(cudaMemcpy_region);

  cudaMemcpy( dev_bucket_caps, bucket_caps, sizeof(int) * THREADS, cudaMemcpyHostToDevice);
  cudaMemcpy( dev_global_splitters, global_splitters, sizeof(int) * (THREADS - 1), cudaMemcpyHostToDevice);

  CALI_MARK_END(cudaMemcpy_region);
  CALI_MARK_END(comm_large);
  CALI_MARK_END(comm);

  CALI_MARK_BEGIN(comp);
  CALI_MARK_BEGIN(comp_large);

  getBucketSize<<<blocks, threads>>>(dev_values, dev_global_splitters, dev_bucket_caps, THREADS, BLOCKS);

  CALI_MARK_END(comp_large);
  CALI_MARK_END(comp);

  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(comm_large);
  CALI_MARK_BEGIN(cudaMemcpy_region);

  cudaMemcpy(bucket_caps, dev_bucket_caps, THREADS * sizeof(int), cudaMemcpyDeviceToHost);

  CALI_MARK_END(cudaMemcpy_region);
  CALI_MARK_END(comm_large);
  CALI_MARK_END(comm);

  // array_print(bucket_caps, THREADS);

  // run prefix sum to calculate starting point

  CALI_MARK_BEGIN(comp);
  CALI_MARK_BEGIN(comp_small);
  bucket_starts[0] = 0;

  for(int i = 1; i < THREADS; i++){
    prefix_sum += bucket_caps[i - 1];
    bucket_starts[i] = prefix_sum;
  }
  CALI_MARK_END(comp_small);
  CALI_MARK_END(comp);

  // array_print(bucket_starts, THREADS);

  // distribute data (use atomic add with size of buckets until offset and size match)
  cudaMalloc((void**) &dev_sorted, NUM_VALS * sizeof(int));
  cudaMalloc((void**) &dev_starts, THREADS * sizeof(int));
  cudaMalloc((void**) &dev_sizes, THREADS* sizeof(int));

  // use the offsets and size to accurately place into sorted data
 
  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(comm_large);
  CALI_MARK_BEGIN(cudaMemcpy_region);

  cudaMemcpy(dev_starts, bucket_starts, THREADS * sizeof(int), cudaMemcpyHostToDevice);

  CALI_MARK_END(cudaMemcpy_region);
  CALI_MARK_END(comm_large);
  CALI_MARK_END(comm);

  // EW TODO: may need to traverse whole array per bucket
  CALI_MARK_BEGIN(comp);
  CALI_MARK_BEGIN(comp_large);

  distributeData<<<blocks, threads>>>(dev_values, dev_sorted, dev_global_splitters, dev_starts, dev_sizes, THREADS, BLOCKS, NUM_VALS);

  CALI_MARK_END(comp_large);
  CALI_MARK_END(comp);

  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(comm_large);
  CALI_MARK_BEGIN(cudaMemcpy_region);

  cudaMemcpy( sorted_array, dev_sorted, NUM_VALS * sizeof(int), cudaMemcpyDeviceToHost);

  CALI_MARK_END(cudaMemcpy_region);
  CALI_MARK_END(comm_large);
  CALI_MARK_END(comm);

  // print sorted array
  // array_print(sorted_array, NUM_VALS);

  /* free device memory */
  cudaFree(dev_values);
  cudaFree(dev_global_splitters);
  cudaFree(dev_bucket_caps);

  CALI_MARK_BEGIN(correctness_check);
  if (!check_array(sorted_array, NUM_VALS)){
    printf("ERROR ARRAY IS NOT SORTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n\n\n");
    sorted = false;
  } else{
    printf("ARRAY IS SORTED\n\n\n");
    sorted = true;
  }
  CALI_MARK_END(correctness_check);

  CALI_MARK_END(main_cali);

  string sort_string;
  string check_string;

  if( sorttype == SORTED)
  {
      sort_string = "sorted";
  } else if( sorttype == PERTURBED){
      sort_string = "1 perturbed";
  } else if( sorttype == REVERSE_SORTED){
      sort_string  = "reversed";
  } else{
      sort_string = "random";
  }

    if(sorted){
        check_string = "success";
    } else{
        check_string = "failure";
    }

  adiak::init(NULL);
  adiak::launchdate();    // launch date of the job
  adiak::libraries();     // Libraries used
  adiak::cmdline();       // Command line used to launch the job
  adiak::clustername();   // Name of the cluster
  adiak::value("Algorithm", "SampleSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
  adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
  adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
  adiak::value("SizeOfDatatype", 4); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
  adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
  adiak::value("InputType", sort_string); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
  adiak::value("num_procs", "0"); // The number of processors (MPI ranks)
  adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
  adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
  adiak::value("group_num", 14); // The number of your group (integer, e.g., 1, 10)
  adiak::value("implementation_source", "Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
  adiak::value("correctness_check", check_string);

  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();

}