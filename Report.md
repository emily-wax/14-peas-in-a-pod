# CSCE 435 Group project

## 1. Group members:
1. Roee Belkin
2. Harini Kumar
3. Ansley Thompson
4. Emily Wax

---

## 2. Team Communication:
our team will be using a slack group chat to communicate. Seeing as we all already use this platform to communicate with the other classmates as well as the TA, it is an easy option for our group. 

## 3. _due 10/25_ Project topic

We will implement 3 parallel sorting algorithms (bubble, quick, and merge sort) in MPI and CUDA. We will examine and compare their performance in detail (computation time, communication time, how much data is sent) on a variety of inputs: sorted, random, reverse, sorted with 1% perturbed, etc.  Strong scaling, weak scaling, GPU performance.

We plan on communicating through Slack.

## 4. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)
- Bubble Sort (MPI + CUDA)
- Quick Sort (MPI + CUDA)
- Merge Sort (MPI + CUDA)

Merge Sort Pseudo Code:

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


Bubble Sort Pseudo Code:

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



References: 
- https://www.geeksforgeeks.org/merge-sort/
- https://www.geeksforgeeks.org/odd-even-sort-brick-sort/
- https://www.geeksforgeeks.org/odd-even-transposition-sort-brick-sort-using-pthreads/
