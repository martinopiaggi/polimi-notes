# Histogram


In the example of computing a histogram, privatization can be applied by having each thread or warp compute **partial** histograms in their own private space in shared or local memory. 

This approach reduces the frequency of atomic updates to global memory, subsequently reducing the number of atomic conflicts. These local histograms are then combined, enhancing performance by decreasing atomic operations/conflicts.

For the histogram example is killer the "privatization" we can privatize at different levels:

- shared memory 
- registers
- Or even committing on different region of global memory to boost performance. Through privatization, each block gets its exclusive region in global memory, reducing conflicts, especially with a large number of bins. This technique enhances efficiency, particularly in scenarios with high memory demand, like when numerous bins are involved.

Obviously the benefits of utilizing shared memory per block and optimizing memory access by committing to shared memory instead of global memory. 

As more transactions are performed with larger inputs, the impact on final performance diminishes. 




The histogram computation with privatization demonstrates this:

```cpp
__global__ void histogram_kernel(const char *__restrict__ data,
                                 unsigned int *__restrict__ histogram,
                                 const unsigned int length) {
  __shared__ unsigned int histo_s[BIN_NUM];
  // Initialize private histogram
  for (unsigned int binIdx = threadIdx.x; binIdx < BIN_NUM; binIdx += blockDim.x) { 
    histo_s[binIdx] = 0u; 
  }
  __syncthreads();

  // Compute histogram using private copy
  for (unsigned int i = tid; i < length; i += stride) {
    const int alphabet_position = data[i] - FIRST_CHAR;
    if (alphabet_position >= 0 && alphabet_position < ALPHABET_SIZE)
      atomicAdd(&(histo_s[alphabet_position / CHAR_PER_BIN]), 1);
  }
  __syncthreads();

  // Commit to global memory
  for (unsigned int binIdx = threadIdx.x; binIdx < BIN_NUM; binIdx += blockDim.x) {
    const unsigned int binValue = histo_s[binIdx];
    if (binValue > 0) {
      atomicAdd(&(histogram[binIdx]), binValue);
    }
  }
}
```

This approach uses a private histogram in shared memory before committing to the global histogram, reducing contention.




In the histogram example, we can see an effort to minimize control divergence:

```cpp
__global__ void histogram_kernel(const char *__restrict__ data,
                                 unsigned int *__restrict__ histogram,
                                 const int length) {
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int stride = blockDim.x * gridDim.x;
  for (unsigned int i = tid; i < length; i += stride) {
    const int alphabet_position = data[i] - FIRST_CHAR;
    if (alphabet_position >= 0 && alphabet_position < ALPHABET_SIZE)
      atomicAdd(&(histogram[alphabet_position / CHAR_PER_BIN]), 1);
  }
}
```
