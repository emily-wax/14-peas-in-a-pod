# CSCE 435 Group project

## 1. Group members:
1. Roee Belkin
2. Harini Kumar
3. Ansley Thompson
4. Emily Wax

---

## 2. _due 10/25_ Project topic

We will implement 3 parallel sorting algorithms (bubble, quick, and merge sort) in MPI and CUDA. We will examine and compare their performance in detail (computation time, communication time, how much data is sent) on a variety of inputs: sorted, random, reverse, sorted with 1% perturbed, etc.  Strong scaling, weak scaling, GPU performance.

We plan on communicating through Slack.

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)
- Bubble Sort (MPI + CUDA)
- Quick Sort (MPI + CUDA)
- Merge Sort (MPI + CUDA)

Merge Sort Pseudo Code:
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

References: 
https://www.geeksforgeeks.org/merge-sort/
