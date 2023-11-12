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

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

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

/* used after sorting is complete, checks if array has 
 * been sucessfully sorted */
bool check_array(float* arr, int length){
  for (int i = 0; i < length -1; i++){
    if (arr[i] > arr[i+1]){
      return false;
    }
  }
  return true; 
}

float random_float(){
  return (float)rand()/(float)RAND_MAX;
}

/* fills array with values depending on desired sort type*/
void array_fill(float *arr, int length, int sort_type){
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

int main(int argc, char *argv[]){

  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  BLOCKS = NUM_VALS / THREADS;

  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of blocks: %d\n", BLOCKS);

  float *values = (float*) malloc( NUM_VALS * sizeof(float));

  array_fill(values, NUM_VALS, RANDOM);

  // EW TODO: put sample sort in

  if (!check_array(values, NUM_VALS)){
    printf("ERROR ARRAY IS NOT SORTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n\n\n");
  }

}