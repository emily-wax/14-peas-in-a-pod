# CSCE 435 Group project

## 1. Group members:
1. Roee Belkin
2. Harini Kumar
3. Ansley Thompson
4. Emily Wax

---

## 2. Team Communication:
Our team will be using a slack group chat to communicate. Seeing as we all already use this platform to communicate with the other classmates as well as the TA, it is an easy option for our group. 

## 3. _due 10/25_ Project topic

We will implement 3 parallel sorting algorithms (bubble, quick, and merge sort) in MPI and CUDA. We will examine and compare their performance in detail (computation time, communication time, how much data is sent) on a variety of inputs: sorted, random, reverse, sorted with 1% perturbed, etc.  Strong scaling, weak scaling, GPU performance.

## 4. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)
- Bubble Sort (MPI + CUDA)
- Quick Sort (MPI + OpenMP)
- Merge Sort (MPI)
- Bitonic Sort (CUDA)

## 3. _due 11/08_ Pseudocode for each algorithm and implementation

**Bubble Sort Pseudo Code:**

Bubble sort in parellel uses a slight variation known as Odd-even or Brick sort.
This odd even variation spreads the data access out slightly so that there can be parellel computation. alternating between 
odd and even ensures that two threads can never be attempting to read/write the same index in the array at the same time. 

```
odd even sort Pseudo Code:

for i from 1 to n-1 with step 2:
    if arr[i] > arr[i + 1]:
	swap arr[i] and arr[i + 1]
	sorted = false


for i from 2 to n-2 with step 2:
    if arr[i] > arr[i + 1]:
	swap arr[i] and arr[i + 1]
	sorted = false

 

1. Initialize MPI and get process rank and size.
2. Determine the total size of the input
3. Calculate the size of each data chunk for each process (chunk_size = ARRAY_SIZE / num_processes).
4. Allocate memory for the local data on each process.
5. If rank == 0:
   a. Initialize the entire array on the root process.
   b. Scatter the array to all processes using MPI_Scatter.
6. Perform the following steps in parallel on each process using CUDA:
   a. Load the local data chunk into GPU memory.
   b. have each thread perform odd-even bubble sort on its set of data
   c. Synchronize the GPU to ensure all threads have completed.
7. Gather the sorted data chunks back to the root process using MPI_Gather.
8. repeat step 6-7 alternating between even and odd bubble sort until all threads merge without making any swaps
9. Merge the sorted data chunks to obtain the final sorted array.
10. repeat the 
11. Finalize MPI.
```

**Quick Sort Pseudo Code:**
```
1. choose a pivot from the unsorted array
2. give each process an even slice of the unsorted array
3. broadcast the pivot out to each process 
4. in each process:
	-Quicksort step: the process's array will be sorted into left <= pivot <= right (happening in parallel)
	-recursively calls quicksort until done
	-lower half of processes send "higher list" to the higher half of processes and vice versa
6. Processes divide into chunks again and the algorithm (step 4) repeats
7. After Log P(num processes) recursions each process i will have lower values than process i+1
8. Sequential quicksort ensues for each process.

Using MPI_Send & Recv as well as Collective Communication (to broadcast the pivot) and create groups of processes.
Number of threads is to be taken in as a variable as well as number of elements in the array.

```

**Merge Sort Pseudo Code:**

```
// merge helper function takes in an array along with left, middle, and right indices
merge(array, left, mid, right):
	// copying from the original array to two new temporary subarrays
  	left_array = array[left to mid]
	right_array = array[mid + 1 to right]	

	left_ptr = 0
	right_ptr = 0
	merged_ptr = left
	
	// merging two subarrays back into the original in sorted order
	while left_ptr < length(left_array) and right_ptr < length(right_array)		
		if left_array[left_ptr] <= right_array[right_ptr]
			array[merged_ptr] = left_array[left_ptr]
			left_ptr += 1
		else
			array[merged_ptr] = left_array[right_ptr]
			right_ptr += 1
		merged_ptr += 1

	while left_ptr < length(left_array)
		array[merged_ptr] = left_array[left_ptr]
		left_ptr += 1
		merged_ptr += 1

	while right_ptr < length(right_array)
		array[merged_ptr] = right_array[right_ptr]
		right_ptr += 1
		merged_ptr += 1	

// merge sort function that recursively calls itself, initially called with left and right as the start and end indices of the array
merge_sort(array, left, right):
	if left >= right
		return

	mid = left + (right - left) / 2
	merge_sort(array, left, mid)
	merge_sort(array, mid + 1, right)
	merge(array, begin, mid, end)
```

Merge sort lends itself well to parallelization since the different recursive calls at the same level work with different parts of the array, so they do not depend on each other (and won’t be written to/read from at the same time). To adapt this algorithm for use with MPI, different MPI processes would each be given a subarray to perform the merge sort algorithm on (using MPI_Scatter), ultimately resulting in an entirely sorted original array. When using CUDA, a similar process would occur using CUDA threads to execute merge sort in parallel on different subsections of the original array on a GPU.

**Bitonic Sort Pseudo Code:**

**References:** 
- https://www.geeksforgeeks.org/merge-sort/
- https://www.geeksforgeeks.org/odd-even-sort-brick-sort/
- https://www.geeksforgeeks.org/odd-even-transposition-sort-brick-sort-using-pthreads/
- http://selkie-macalester.org/csinparallel/modules/MPIProgramming/build/html/mergeSort/mergeSort.html

## 3. _due 11/08_ Evaluation plan - what and how will you measure and compare
- Strong scaling to more nodes (same problem size, increase number of processors)
    - For each algorithm and each problem size, graph average runtime (across input types) vs number of threads
    - Multiple lines per graph representing timings for different sections (computation time, communication time, etc.)
    - Identify and compare at which point adding more threads does not result in further speedup for each algorithm and problem size
- Compare sorting algorithms’ parallel performance on different input types (sorted, random, reverse, sorted with 1% perturbed)
    - Bar graph comparing overall runtimes for each input type for the largest chosen problem size
    - Identify which input type(s) each algorithm performs best on
    - Compare overall parallel performance across algorithms using MPI/CUDA
 
Runtimes will be recorded using Caliper regions (separating the timings for data generation, computation, communication, and correctness checking).

## 3. Project implementation
Implement your proposed algorithms, and test them starting on a small scale.
Instrument your code, and turn in at least one Caliper file per algorithm;
if you have implemented an MPI and a CUDA version of your algorithm,
turn in a Caliper file for each.

### 3a. Caliper instrumentation
Please use the caliper build `/scratch/group/csce435-f23/Caliper/caliper/share/cmake/caliper` 
(same as lab1 build.sh) to collect caliper files for each experiment you run.

Your Caliper regions should resemble the following calltree
(use `Thicket.tree()` to see the calltree collected on your runs):
```
main
|_ data_init
|_ comm
|    |_ MPI_Barrier
|    |_ comm_small  // When you broadcast just a few elements, such as splitters in Sample sort
|    |   |_ MPI_Bcast
|    |   |_ MPI_Send
|    |   |_ cudaMemcpy
|    |_ comm_large  // When you send all of the data the process has
|        |_ MPI_Send
|        |_ MPI_Bcast
|        |_ cudaMemcpy
|_ comp
|    |_ comp_small  // When you perform the computation on a small number of elements, such as sorting the splitters in Sample sort
|    |_ comp_large  // When you perform the computation on all of the data the process has, such as sorting all local elements
|_ correctness_check
```

Required code regions:
- `main` - top-level main function.
    - `data_init` - the function where input data is generated or read in from file.
    - `correctness_check` - function for checking the correctness of the algorithm output (e.g., checking if the resulting data is sorted).
    - `comm` - All communication-related functions in your algorithm should be nested under the `comm` region.
      - Inside the `comm` region, you should create regions to indicate how much data you are communicating (i.e., `comm_small` if you are sending or broadcasting a few values, `comm_large` if you are sending all of your local values).
      - Notice that auxillary functions like MPI_init are not under here.
    - `comp` - All computation functions within your algorithm should be nested under the `comp` region.
      - Inside the `comp` region, you should create regions to indicate how much data you are computing on (i.e., `comp_small` if you are sorting a few values like the splitters, `comp_large` if you are sorting values in the array).
      - Notice that auxillary functions like data_init are not under here.

All functions will be called from `main` and most will be grouped under either `comm` or `comp` regions, representing communication and computation, respectively. You should be timing as many significant functions in your code as possible. **Do not** time print statements or other insignificant operations that may skew the performance measurements.

**Nesting Code Regions** - all computation code regions should be nested in the "comp" parent code region as following:
```
CALI_MARK_BEGIN("comp");
CALI_MARK_BEGIN("comp_large");
mergesort();
CALI_MARK_END("comp_large");
CALI_MARK_END("comp");
```

**Looped GPU kernels** - to time GPU kernels in a loop:
```
### Bitonic sort example.
int count = 1;
CALI_MARK_BEGIN("comp");
CALI_MARK_BEGIN("comp_large");
int j, k;
/* Major step */
for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
        bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
        count++;
    }
}
CALI_MARK_END("comp_large");
CALI_MARK_END("comp");
```

**Calltree Examples**:

```
# Bitonic sort tree - CUDA looped kernel
1.000 main
├─ 1.000 comm
│  └─ 1.000 comm_large
│     └─ 1.000 cudaMemcpy
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

```
# Matrix multiplication example - MPI
1.000 main
├─ 1.000 comm
│  ├─ 1.000 MPI_Barrier
│  ├─ 1.000 comm_large
│  │  ├─ 1.000 MPI_Recv
│  │  └─ 1.000 MPI_Send
│  └─ 1.000 comm_small
│     ├─ 1.000 MPI_Recv
│     └─ 1.000 MPI_Send
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

```
# Mergesort - MPI
1.000 main
├─ 1.000 comm
│  ├─ 1.000 MPI_Barrier
│  └─ 1.000 comm_large
│     ├─ 1.000 MPI_Gather
│     └─ 1.000 MPI_Scatter
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

#### 3b. Collect Metadata

Have the following `adiak` code in your programs to collect metadata:
```
adiak::init(NULL);
adiak::launchdate();    // launch date of the job
adiak::libraries();     // Libraries used
adiak::cmdline();       // Command line used to launch the job
adiak::clustername();   // Name of the cluster
adiak::value("Algorithm", algorithm); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
adiak::value("ProgrammingModel", programmingModel); // e.g., "MPI", "CUDA", "MPIwithCUDA"
adiak::value("Datatype", datatype); // The datatype of input elements (e.g., double, int, float)
adiak::value("SizeOfDatatype", sizeOfDatatype); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
adiak::value("implementation_source", implementation_source) // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
```

They will show up in the `Thicket.metadata` if the caliper file is read into Thicket.

**See the `Builds/` directory to find the correct Caliper configurations to get the above metrics for CUDA, MPI, or OpenMP programs.** They will show up in the `Thicket.dataframe` when the Caliper file is read into Thicket.
