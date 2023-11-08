# CSCE 435 Group project
## 0. Group number: 14
## 1. Group members:
1. Roee Belkin
2. Harini Kumar
3. Ansley Thompson
4. Emily Wax

---

## 2. Team Communication:
Our team will be using a slack group chat to communicate. Seeing as we all already use this platform to communicate with the other classmates as well as the TA, it is an easy option for our group. 

## 3. _due 10/25_ Project topic

We will implement 3 parallel sorting algorithms (bubble, sample, and merge sort) in MPI and CUDA. We will examine and compare their performance in detail (computation time, communication time, how much data is sent) on a variety of inputs: sorted, random, reverse, sorted with 1% perturbed, etc.  Strong scaling, weak scaling, GPU performance.

## 4. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)
- Bubble Sort (MPI + CUDA)
- Sample Sort (MPI)
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

**Sample Sort Pseudo Code:**
```
1. Set up MPi Threads/Generate Data
2. Choose (p-1) local splitters in each process (evenly separated)
3. Gather all local splitters into root function (global splitters list) (MPI_Gather)
4. Sort new "global splitters" list
5. Narrow down to p-1 global splitters (evenly separated)
6. Broadcast global splitters to all processes (MPI_Bcast)
7. Perform bucket sort in each process using global splitters as indices - use one vector per bucket
8. Each process sends buckets to other processes (based on bucket indices) (MPI_Send)
9. Each process receives values in its bucket from other processes (MPI_Recv & MPI_Get_Count)
	a. receive values into "final bucket" vector, using MPI_Get_Count to size appropriately
10. Local sort is performed on each process
11. Check if sorted
```

MPI Calls used

**Merge Sort Pseudo Code:**

```
mergesort(tree_height, thread_id, thread_array, arr_size, global_array):
	curr_height = 0
	left_data = thread_array
	initialize right_data, merged_data as nullptr

	// tree_height refers to height of merge tree (starting from one sorted array for each thread at height 0 to one fully merged array at tree_height)
	// at each height, adjacent (left and right) processes are merged together into the left process
	while curr_height < tree_height:
		if is_left_branch:
			find corresponding right_branch thread id
			
			MPI_RECV data from right_branch process to right_data

			merged_data = results of calling merge function on left_data and right_data

			left_data = merged_data // since left branch is one that will continue working
			double arr_size, handle memory updates, increment curr_height in preparation for next loop iteration

		else: // in right branch
			find corresponding left_branch thread id
			
			MPI_Send this process's data (currently held in left_data) to left_branch

			handle memory, update curr_height to tree_height to break out of while loop 

			// while loop terminates for right branches, number of active threads halves at each level of tree
			
	for root thread, set global_array equal to left_data (now holds final merged sorted array)

main:
	take in num_threads and num_vals (to sort) as input
	set up MPI: MPI_Init, MPI_Comm_size, MPI_Comm_rank

	block_size = num_vals / num_threads

	in root thread, allocate values_array_global

	call createData function in all threads to generate data to sort in parallel

	MPI_Scatter from the values_array_global to values_array_thread (one per thread, each of length block size)
	
	call sequential_mergesort function on each thread, resulting in every thread having a sorted values_array_thread

	call mergesort function on all threads, pass in log2(num_threads) as tree_height, block_size as arr_size

	in root thread, call correctness_check function to ensure that values_array_global is sorted
```

**Bitonic Sort Pseudo Code:**
The code from our Lab 3 implementation will be used for bitonic sort.

```
bitonic_sort:
	for the length of the array:
		sort first half ascending
		sort second half descendings
		call merge on different directions with different threads
		merge in the given direction
merge:
	for the length of the array:
		swap arr[i] and arr[i+n/2] if in wrong order for direction
	merge first half, recursive
	merge second half, recursive
```

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
### Status:

**Bubble Sort:**
The MPI implementation of bubble sort is complete, I have sources for the [CUDA implementation]([url](https://github.com/domkris/CUDA-Bubble-Sort/blob/master/CUDABubbleSort/kernel.cu)) in a couple different ways but have not implemented it yet.

**Sample Sort:**
The MPI implementation of sample sort is complete. Research has been done for the CUDA version but it has not been implemented yet.

**Merge Sort:**
The MPI implementation of merge sort is complete. I have found a [source]([url](https://pushkar2196.wordpress.com/2017/04/19/mergesort-cuda-implementation/#:~:text=GPUs%20are%20optimized%20for%20running,threads%20at%20the%20same%20time.&text=Merge%20Sort%20is%20a%20Divide,used%20for%20merging%20two%20halves.)) to adapt for the CUDA implementation, but have not implemented it yet.

**Bitonic Sort:**
The CUDA implementation of bitonic sort is complete, I have found code for the [MPI implementation]([url](https://people.cs.rutgers.edu/~venugopa/parallel_summer2012/mpi_bitonic.html)https://people.cs.rutgers.edu/~venugopa/parallel_summer2012/mpi_bitonic.html) but have not implemented it yet.

## Questions:
* How important is it to generate data in parallel for CUDA? Since Lab 3 did not, we also did not for our CUDA implementation.
