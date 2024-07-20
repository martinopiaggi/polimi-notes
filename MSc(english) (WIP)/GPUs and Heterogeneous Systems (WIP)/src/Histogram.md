# Histogram

A histogram is a display of the number count of occurrences of data values in a dataset.
Often the data items are grouped into specific ranges or bins  (e.g. a specific color in an image, a letter (or group of letters) in a text). 
They are used whenever there is a large volume of data that needs to be analyzed to distill interesting events (think of feature extraction in computer vision).
Parallelization Strategy:

- Each thread processes a portion of the input array.
- Atomic operations are used to update the global histogram to avoid race conditions.

```cpp
__global__ void histogram_kernel(const char *__restrict__ data,
                                 unsigned int *__restrict__ histogram,
                                 const unsigned int length) {
const int i = threadIdx.x + blockIdx.x * blockDim.x;

if (i < length) {
    const int alphabet_position = data[i] - FIRST_CHAR;
    if (alphabet_position >= 0 && alphabet_position < ALPHABET_SIZE)
      atomicAdd(&(histogram[alphabet_position / CHAR_PER_BIN]), 1);
}
```

## Optimizations

### Coarsening

Each thread processes multiple elements.
```cpp
__global__ void
    histogram_kernel(const char *__restrict__ data, unsigned int *__restrict__ histogram, const int length) {
  const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int stride = blockDim.x * gridDim.x;
  
  // All threads in a blockk handle consecutive elements in each iteration
  for (unsigned int i = tid; i < length; i += stride) {
    const int alphabet_position = data[i] - FIRST_CHAR;
    if (alphabet_position >= 0 && alphabet_position < ALPHABET_SIZE)
      atomicAdd(&(histogram[alphabet_position / CHAR_PER_BIN]), 1);
  }
}
```
### Privatization

Each thread maintains a local histogram before merging into the global one. We can privatize at different levels:

- shared memory 
- registers
- Or even committing on different region of global memory to boost performance. 

```cpp
__global__ void histogram_kernel(const char *__restrict__ data,
                                 unsigned int *__restrict__ histogram,
                                 const unsigned int length) {
  
  const unsigned int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int stride = blockDim.x * gridDim.x;
  // Privatized bins
  __shared__ unsigned int histo_s[BIN_NUM];
#pragma unroll
  for (unsigned int binIdx = threadIdx.x; binIdx < BIN_NUM; binIdx += blockDim.x) { histo_s[binIdx] = 0; }
  __syncthreads();
  // Histogram
  for (unsigned int i = tid; i < length; i += stride) {
    const int alphabet_position = data[i] - FIRST_CHAR;
    if (alphabet_position >= 0 && alphabet_position < ALPHABET_SIZE)
      atomicAdd(&(histo_s[alphabet_position / CHAR_PER_BIN]), 1);
  }
  __syncthreads();
// Commit to global memory
#pragma unroll
  for (unsigned int binIdx = threadIdx.x; binIdx < BIN_NUM; binIdx += blockDim.x) {
    const unsigned int binValue = histo_s[binIdx];
    if (binValue > 0) {
      atomicAdd(&(histogram[binIdx]), binValue);
    }
  }
}
```

This approach uses a private histogram in shared memory before committing to the global histogram, reducing contention.

```cpp
__global__ void histogram_kernel(const char *__restrict__ data,
                                 unsigned int *__restrict__ histogram,
                                 const unsigned int length) {
  const unsigned int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int stride = blockDim.x * gridDim.x;
  
  // Privatized bins
  unsigned int histo_p[BIN_NUM];
  
__shared__ unsigned int histo_s[BIN_NUM];
#pragma unroll
for (unsigned int i = threadIdx.x; i < BIN_NUM; i += blockDim.x) {
    histo_s[i] = 0;
}
__syncthreads();

#pragma unroll
  for (unsigned int i = 0; i < BIN_NUM; i++) histo_p[i] = 0u;
  // Histogram
  for (unsigned int i = tid; i < length; i += stride) {
    const int alphabet_position = data[i] - FIRST_CHAR;
    if (alphabet_position >= 0 && alphabet_position < ALPHABET_SIZE)
      histo_p[alphabet_position / CHAR_PER_BIN] += 1;
  }
  // Commit to shared memory
#pragma unroll
  for (unsigned int binIdx = 0; binIdx < BIN_NUM; binIdx++) {
    const unsigned int binValue = histo_p[binIdx];
    if (binValue > 0) {
      atomicAdd(&(histo_s[binIdx]), binValue);
    }
  }

__syncthreads(); // Synchronization barrier

// Commit to global memory
#pragma unroll
  for (unsigned int binIdx = threadIdx.x; binIdx < BIN_NUM; binIdx += blockDim.x) {
    const unsigned int binValue = histo_s[binIdx];
    if (binValue > 0) {
      atomicAdd(&(histogram[binIdx]), binValue);
    }
  }
}
```
### Aggregation

The concept of aggregating data values in contiguous regions to reduce atomic operations can be beneficial.
To reduce high contention consecutive updates to the same bin are combined.

```cpp
__global__ void histogram_kernel(const char *__restrict__ data,
                                 unsigned int *__restrict__ histogram,
                                 const unsigned int length) {
  const unsigned int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int stride = blockDim.x * gridDim.x;
  // Privatized bins
  __shared__ unsigned int histo_s[BIN_NUM];
#pragma unroll
  for (unsigned int binIdx = threadIdx.x; binIdx < BIN_NUM; binIdx += blockDim.x) { histo_s[binIdx] = 0u; }
  __syncthreads();
  
  // Histogram
  unsigned int accumulator = 0;
  int prevBinIdx           = -1;
  for (unsigned int i = tid; i < length; i += stride) {
    int alphabet_position = data[i] - FIRST_CHAR;
    if (alphabet_position >= 0 && alphabet_position < ALPHABET_SIZE) {
      const int bin = alphabet_position / CHAR_PER_BIN;
      if (bin == prevBinIdx) {
        ++accumulator;
      } else {
        if (accumulator > 0) {
          atomicAdd(&(histo_s[prevBinIdx]), accumulator);
        }
        accumulator = 1;
        prevBinIdx  = bin;
      }
    }
  }
  if (accumulator > 0) {
    atomicAdd(&(histo_s[prevBinIdx]), accumulator);
  }
  __syncthreads();
// Commit to global memory
#pragma unroll
  for (unsigned int binIdx = threadIdx.x; binIdx < BIN_NUM; binIdx += blockDim.x) {
    const unsigned int binValue = histo_s[binIdx];
    if (binValue > 0) {
      atomicAdd(&(histogram[binIdx]), binValue);
    }
  }
}
```