

# Convolution, stencil, histogram, reduction, prefix sum, merge


## Convolution 

[Convolutional Neural Networks](projects/polimi-notes/MSc(english)%20(WIP)/Artificial%20Neural%20Networks%20and%20Deep%20Learning%20(WIP)/src/03.Image%20Classification.md#Convolutional%20Neural%20Networks) are based on the convolution operation: multiply the kernel with the input matrix elements, and sum them up to write to the output matrix. 

```cpp
void convolution_cpu(input_type *input, const input_type *filter, input_type *output, int width, int height, int filter_size, int filter_radius) {

    // Iterate over each output pixel
    for (int outRow = 0; outRow < height; outRow++) {
        for (int outCol = 0; outCol < width; outCol++) {
            input_type value = 0.0f;  
            
            // Apply the filter 
            for (int row = 0; row < filter_size; row++) {
                for (int col = 0; col < filter_size; col++) {
                    
                    int inRow = outRow - filter_radius + row; 
                    int inCol = outCol - filter_radius + col;
                    
                    if (inRow >= 0 
	                && inRow < height 
	                && inCol >= 0 
	                && inCol < width) {
	                    value += 
                        filter[row * filter_size + col] * 
                        input[inRow * width + inCol];
                    }
                }
            }
            output[outRow * width + outCol] = value;
        }
    }
}
```

### CONV2D Computing

The convolution kernel benefits from using constant memory, which is optimized for situations where all threads read the same value, such as filter coefficients in convolution operations.

- **Serial Access:** Accessing the same constant memory location on each clock cycle significantly speeds up the operation since it avoids the latency of global memory access.
- **Memory Initialization:** The `__constant__` memory space is declared at the global level and can be accessed across multiple kernel launches without reinitialization.

Define symbols in the same file as the kernel to ensure they are recognized by the linker. Use the `extern` keyword for symbols declared in other compilation units.

```cpp
// Copy filter to constant memory
cudaMemcpyToSymbol(filter, hostFilter, sizeof(float) * KERNEL_SIZE);
//or more explicitly 
cudaMemcpyToSymbol(constant_filter, filter, FILTER_SIZE * FILTER_SIZE * sizeof(filter_type))
```

- **Symbolic Filter:** By treating the filter as a symbol rather than a variable, the kernel call simplifies as the filter does not need to be passed as a parameter.
- **Loop Unrolling:** With a fixed filter size, the compiler can optimize the convolution operation by unrolling loops, enhancing execution efficiency.


```cpp
// GPU filter for convolution CONSTANT MEMORY
__global__ void convolution_constant_mem_kernel(const input_type *__restrict__ input, input_type *__restrict__ output, const int width, const int height) {
  
  const int outCol = blockIdx.x * blockDim.x + threadIdx.x;
  const int outRow = blockIdx.y * blockDim.y + threadIdx.y;
  input_type value{0.0f};

#pragma unroll
  for (int row = 0; row < FILTER_SIZE; row++)

#pragma unroll
    for (int col = 0; col < FILTER_SIZE; col++) {
      const int inRow = outRow - FILTER_RADIUS + row;
      const int inCol = outCol - FILTER_RADIUS + col;
		if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
	        value += constant_filter[row][col] * input[inRow * width + inCol];
	    }
    }
	output[outRow * width + outCol] = value;
}
```

### Coarsening 


Coarsening: each thread -> perform more work . 

1. **Increased Arithmetic Intensity:** By allowing each thread to perform more operations, the ratio of compute operations to memory operations increases, improving the overall performance on compute-bound tasks.
2. **Reduced Launch Overhead:** Fewer threads mean less overhead in thread scheduling and management on the GPU.
3. **Improved Memory Access Patterns:** Coarsening can lead to more coalesced memory accesses if the threads access adjacent memory locations, which is optimal for GPUs.
4. **Better Resource Utilization:** Maximizing the work each thread does can lead to more efficient use of the GPU's cores, which might otherwise be underutilized in a finely-grained parallel scheme.

In the context of convolution operations, coarsening can be applied by having each thread compute multiple output elements instead of one. Below is a simplified example of how you might implement a coarsened approach in a GPU kernel for a convolution operation:

```cpp
__global__ void convolution_constant_mem_coarsening_kernel(const input_type *__restrict__ input,input_type *__restrict__ output,const int width,const int height) {

  // Calculate the starting column and row for the current thread
  const int outCol = blockIdx.x * blockDim.x + threadIdx.x;
  const int outRow = blockIdx.y * blockDim.y + threadIdx.y;

  // Calculate the strides for columns and rows based on the grid dimensions
  const int stride_col = gridDim.x * blockDim.x;
  const int stride_row = gridDim.y * blockDim.y;

  // Loop over assigned columns and rows using stride values
  for (int c = outCol; c < width; c += stride_col) {
    for (int r = outRow; r < height; r += stride_row) {
      input_type value = 0.0f; 
      #pragma unroll
      for (int row = 0; row < FILTER_SIZE; row++) {
        #pragma unroll
        for (int col = 0; col < FILTER_SIZE; col++) {
          const int inRow = r - FILTER_RADIUS + row; // Compute input row index
          const int inCol = c - FILTER_RADIUS + col; // Compute input column index

          // Check boundary conditions and perform convolution
          if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
            value += constant_filter[row][col] * input[inRow * width + inCol];
          }
        }
      }
      output[r * width + c] = value; // Store the computed value in the output image
    }
  }
}

```

- **Coarsening Factor:** Each thread computes a 2x2 block of the output image, effectively coarsening the thread's workload by a factor of four.
- **Boundary Conditions:** The code checks whether each output pixel location falls within the image boundaries before performing the convolution.
- **Nested Loops for Convolution:** The actual convolution computation involves nested loops over the filter dimensions and applies the filter to the relevant part of the input image.


`stride_col` and `stride_row` to skip across the image based on the grid size. The stride depends on the number of spawned threads. 


**Shared Memory Optimization Techniques**

Using shared memory in this way minimizes the reliance on global memory and speeds up the convolution by allowing faster access to input data.

Using a shared memory approach with tiling significantly alters how data is accessed and processed on a GPU, emphasizing efficiency and reduced global memory dependency. This method is particularly effective for operations like convolution where data reuse is high. The approach minimizes the latency of data access and maximizes the throughput by leveraging the fast shared memory on the GPU.

**First Load the Tiles:** Each thread in a block loads an element of the input image into shared memory, including necessary padding for the convolution operation. This reduces global memory accesses. After loading the data into shared memory, threads are synchronized to ensure all data is properly loaded before computation begins.
   ```cpp
   __shared__ input_type input_shared[IN_TILE_DIM][IN_TILE_DIM];
   if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
     input_shared[tidy][tidx] = input[inRow * width + inCol];
   } else {
     input_shared[tidy][tidx] = 0.0;
   }
   __syncthreads();
   ```

**Compute Output Elements:** Each thread computes an element of the output using the data in shared memory. This computation is localized to the data loaded by the block, significantly reducing the latency associated with memory access.

   ```cpp
   const int tileCol = tidx - FILTER_RADIUS
   const int tileRow = tidy - FILTER_RADIUS
   
   if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM) {
     input_type output_value{0.0f};
     #pragma unroll
     for (int row = 0; row < FILTER_SIZE; row++)
     #pragma unroll
       for (int col = 0; col < FILTER_SIZE; col++)
         output_value += constant_filter[row][col] * input_shared[tileRow + row][tileCol + col];
     output[inRow * width + inCol] = output_value;
   }
   ```



### Caching Halo cells

![](images/Pasted%20image%2020240423190143.png)

During execution, only the internal part of the tile is loaded into shared memory. The assumption is that halo cells are potentially available in L2 cache, having been loaded by adjacent blocks in previous operations.

In the inner loop:

```cpp
if (sharedRow >= 0 && sharedRow < TILE_DIM 
	&& sharedCol >= 0 && sharedCol < TILE_DIM) {
		//shared part
		PValue += constant_filter[fRow][fCol] * input_shared[sharedRow][sharedCol];
} else {
	// Global memory
	int globalRow = row - FILTER_RADIUS + fRow;
	int globalCol = col - FILTER_RADIUS + fCol;
	if (globalRow >= 0 && globalRow < height 
		&& globalCol >= 0 && globalCol < width) {
		PValue += constant_filter[fRow][fCol] 
			* input[globalRow * width + globalCol];
	}
}
```


Due to conditional statements handling halo and boundary conditions there is a **a lot of Branch Divergence**, which increase instruction count and reduce performance.
**Compiler's Role:** the compiler generates a substantial number of instructions to manage this complexity, which can aggravate thread divergence.


## Reduction 

The concept of reduction involves collecting values from an array or collection and reducing them to a single value. Examples are the sum or finding the maximum value in an array. 

The naive GPU implementation that divides the input into blocks of threads and has each thread process two elements. The threads then combine their results until only one thread remains, which performs the final reduction operation. 

![](images/Pasted%20image%2020240501180528.png)

In this case the stride factor is two Essentially, it defines how far apart the memory locations each thread should read from or write to are spaced in each step of the reduction. In each subsequent step of the reduction process, the stride factor typically doubles, ensuring that each thread combines data from increasingly distant parts of the input array. This helps to efficiently reduce large arrays by combining results across the threads in a block, and eventually across multiple blocks, to arrive at a final result.

The value "two" you mentioned is likely the initial value for the stride factor, which indicates that initially, each thread reads data separated by one intervening element. The stride then doubles with each iteration of the reduction loop, allowing threads to progressively cover and reduce larger portions of data.

```cpp
// CPU version of the reduction kernel
float reduce_cpu(const float* data, const int length) {
  float sum = 0;
  for (int i = 0; i < length; i++) { sum += data[i]; }
  return sum;
}
```
Each thread within a block performs a computation, accumulates values, and exchanges results with neighboring threads before storing it in global memory. The speaker highlights the importance of shifting input data between blocks using the stride factor and block index, ensuring each block processes different parts of the input.

The output will be a vector of double with the same dimension as the grid dimension, which is calculated by dividing the input dimension by the number of elements processed by each block. In the end, only the first thread of each block will have the final result after performing the reduction from all elements.

```cpp
// GPU version of the reduction kernel
__global__ void reduce_gpu(double* __restrict__ input, double* __restrict__ output) {
  const unsigned int i = STRIDE_FACTOR * threadIdx.x;

  // Apply the offset
  input += blockDim.x * blockIdx.x * STRIDE_FACTOR;
  output += blockIdx.x;

  for (unsigned int stride = 1; stride <= blockDim.x; stride *= STRIDE_FACTOR) {
    if (threadIdx.x % stride == 0) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }

  // Write result for this block to global memory
  if (threadIdx.x == 0) {
    // You could have used only a single memory location and performed an atomicAdd
    *output = input[0];
  }
}
```



- `if (threadIdx.x % stride == 0)`: This condition checks if the current thread should participate in this iteration of the reduction. A thread participates if its index is a multiple of the current `stride`, meaning it is responsible for adding a specific element located `stride` positions away.

This current code relies heavily on global memory access with minimal use of cache levels. 



Exploring ways to reduce memory transfer and improve memory locality is necessary. 

### Memory divergency 

The key change is using the stride index in reverse order, dividing it by 2 at each iteration to maintain the same approach as before. An if condition is used to select which strides are active and which are not, based on the triad index being smaller than the actual value of the strides. This ensures that only necessary memory locations are accessed.

![](images/Pasted%20image%2020240501182020.png)

```cpp
// Coalesced
__global__ void reduce_gpu(double* __restrict__ input, double* __restrict__ output) {
  const unsigned int i = threadIdx.x;

  // Apply the offset
  input += blockDim.x * blockIdx.x*STRIDE_FACTOR;
  output += blockIdx.x;

  for (unsigned int stride = blockDim.x; stride >= 1; stride /= STRIDE_FACTOR) {
    if (i < stride) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }

  // Write result for this block to global memory
  if (threadIdx.x == 0) {
    // You could have used only a single memory location and performed an atomicAdd
    *output = input[0];
  }
}
```


Coalesced Access -> Each thread in a warp accesses memory locations that are consecutive -> if thread 0 accesses memory address X, thread 1 accesses address X+1, thread 2 accesses X+2, and so on -> When threads in a warp access consecutive memory addresses, the GPU can combine these accesses into a single memory transaction -> 1. **Reduced Memory Transactions** ->  **Improved Bandwidth Utilization** -> **Lower Latency**


### Privatization

Privatization as another optimization technique, where instead of going back and forth between global memory, computations are performed in shared memory.

Threads write their partial sum result values to global memory
o These values are reread by the same threads and others in the next iterations
Shared memory has much shorter latency and higher bandwidth
o It can fit the data required by the block's thread in this case

```cpp
// Privatization 
__global__ void reduce_gpu(const double* __restrict__ input, double* __restrict__ output) {
  __shared__ double partial[BLOCK_DIM]; 
  const unsigned int i = threadIdx.x;

  // Apply the offset
  input += blockDim.x * blockIdx.x*STRIDE_FACTOR;
  output += blockIdx.x;

  partial[i] = input[i] + input[i+BLOCK_DIM] //first iteration outside and LOAD

  for (unsigned int stride = blockDim.x / STRIDE_FACTOR; stride >= 1; stride /= STRIDE_FACTOR) {
    if (i < stride) {
      partial[i] += partial[i + stride];
    }
    __syncthreads();
  }

  // Write result for this block to global memory
  if (threadIdx.x == 0) {
    // You could have used only a single memory location and performed an atomicAdd
    *output = partial[0];
  }
}
```

The use of shared memory allows for faster access times compared to global memory. 

the first step of the reduction before entering the loop. This step effectively halves the number of active threads needed in the subsequent steps by pre-summing two distant elements of the input array. This is a common technique in reduction kernels where the first step is explicitly handled to reduce the complexity of the loop and decrease the number of iterations required.



To maximize memory bandwith and reducing workload (fewer iterations and threads) the initial step where data is copied from global to shared memory is not just about moving data around but an opportune moment to perform an initial partial sum of the data.


### Thread coarsening 

To reduce overhead, thread coarsening can be employed, allowing for more efficient use of hardware resources. 
Actually two solutions depending on the `COARSE_FACTOR`:


![Image illustrating trade-offs and performance considerations](images/Pasted%20image%2020240501184452.png)


Regarding spawning fewer threads: 

![Image showing detailed thread processing](images/Pasted%20image%2020240501184314.png)

 The method involves serializing these thread blocks ourselves in a more efficient manner than the hardware would. The segment for each block is identified by multiplying with the COARSE FACTOR, and threads are tasked with more than just adding two elements.

  - Threads begin by calculating their own unique starting point in the input array by adjusting for block and thread indices multiplied by the `COARSE_FACTOR`.
  - Each thread sums multiple elements from the input array, determined by the `COARSE_FACTOR`
  - This sum is initially calculated using regular registers due to their faster access times compared to shared memory, although this approach increases register pressure which can potentially reduce parallelism.
  - Subsequent reduction of these sums is performed in shared memory, consolidating the data further through synchronized strides, reducing the number of active threads in each step, and enhancing memory access patterns within the GPU's architecture.


```cpp
// GPU version of the reduction kernel
__global__ void reduce_coarsening_gpu(const double* __restrict__ input, double* __restrict__ output) {
  __shared__ double input_s[BLOCK_DIM];
  const unsigned int t = threadIdx.x;

  // Apply the offset
  // NOTE: input const means that the content is const, the pointer can change
  input += blockDim.x * blockIdx.x * COARSE_FACTOR;
  output += blockIdx.x;

  double sum = input[t];
  // Here the hardware is fully utilized
  for (unsigned int tile = 1; tile < COARSE_FACTOR; ++tile) sum += input[t + tile * BLOCK_DIM];
  // You could have used directly the shared memory
  // Registers are faster though
  // High register pressure leads to lower parallelism
  input_s[t] = sum;

  // Perform reduction in shared memory
  for (unsigned int stride = blockDim.x / STRIDE_FACTOR; stride >= 1; stride /= STRIDE_FACTOR) {
    __syncthreads();
    if (t < stride) {
      input_s[t] += input_s[t + stride];
    }
  }

  // Write result for this block to global memory
  if (threadIdx.x == 0) {
    // You could have used only a single memory location and performed an atomicAdd
    *output = input_s[0];
  }
}
```

Key takeaways include the significant impact of adjusting the coarse factor and block size on performance, the benefits of using shared memory for accessing input elements which enhances parallel processing efficiency, and the optimization of reduction operations in global memory through atomic operations or synchronization mechanisms. The balance between high register usage and memory efficiency is crucial for optimizing GPU kernel performance.


## Scan 

A scan operation, also known as a prefix sum, is similar to reduction but produces an array of results instead of a single value.

![](images/Pasted%20image%2020240501191013.png)

**Two types of scan:

- **Inclusive scan**: includes current element in partial reduction. 
- **Exclusive scan**: excludes current element in partial reduction, partial reduction is of all prior elements prior to current element.

The choice between inclusive and exclusive scans can affect how the parallelization strategy is implemented due to the initial value settings and propagation of sum values through the dataset.

```cpp

// CPU version of the scan kernel EXCLUSIVE
void scan_cpu(const float* input, float* output, const int length) {
  output[0] = 0;
  for (int i = 1; i < length; ++i) { output[i] = output[i - 1] + input[i - 1]; }
}

// Difference with reduction operation:
float reduce_cpu(const float* data, const int length) {
  float sum = 0;
  for (int i = 0; i < length; i++) { sum += data[i]; }
  return sum;
}
```

There is the risk to try to parallelize a sequential version which has the bottleneck of the longest path (leading to $O(n^2)$): each thread has to look at all previous elements, leading to redundant work and poor utilization of the GPU's algorithmic capabilities: 

```cpp
__global__ void
    naive_scan_gpu(const float* __restrict__ input, float* __restrict__ output, const int length) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  float sum{0};
  for (unsigned int i = 0; i < index; i++) { sum += input[i]; }
  output[index] = sum;
  return;
}
```


The Kogge-Stone algorithm is used for efficiently summing elements in parallel on a GPU.
Each thread sums a range of elements and then combines these sums with results from other threads. *Basically I increase the parallelization with a tradeoff of synchronization.*


![](images/Pasted%20image%2020240508212724.png)

```cpp
// GPU version of the scan kernel EXCLUSIVE Kogge Stone
__global__ void
    kogge_stone_scan_gpu(const float* __restrict__ input, float* __restrict__ output, const int length) {
  __shared__ float input_s[BLOCK_DIM];
  const unsigned int tid = threadIdx.x;

  input_s[tid] = input[tid - 1];
  
  for (unsigned int stride = 1; stride < length; stride *= STRIDE_FACTOR) {
    __syncthreads();
    float temp;
    if (tid >= stride)
      temp = input_s[tid] + input_s[tid - stride];
    __syncthreads();
    if (tid >= stride)
      input_s[tid] = temp;
  }
  output[tid] = input_s[tid];

  return;
}
```


- **Initial State**: Each `output[i]` starts with the corresponding input element x.
- **Iterations**: After `k` iterations, `output[i]` will contain the sum of `2^k` input elements.
- **Thread ID Adjustment**: `input_s[tid] = input[tid - 1];` initializes each thread with the input shifted by one index back. This is critical for managing edge cases where `tid = 0`.
- **Loop for Summation**: The loop increases the `stride` geometrically, reducing the number of active threads at each step. This reduction is typical in parallel reduction algorithms to minimize active threads as the problem size effectively decreases.
- **Synchronization**: Twice per loop iteration to avoid hazards:
  - Before calculating the temporary sum to ensure all threads have the correct values (`__syncthreads()`).
  - After storing the temporary sum to ensure that no writes occur until all calculations in the current stride are complete (`__syncthreads()`).


note that in this version - All threads must wait at synchronization points to ensure data integrity, which can introduce delays or "stalls" where threads are idle, waiting for others to reach the synchronization point.


Another version is the improved Kogge-Stone algorithm using a double buffering technique: double the shared memory for each block to alternate between reading input and writing output across iterations. This technique, known as "ping-pong", alternates the roles of input and output buffers, reducing the need for synchronization barriers typically required in single-buffer implementations.

```cpp
// Double buffering approach scan
__global__ void
    kogge_stone_scan_gpu(const float* __restrict__ input, float* __restrict__ output, const int length) {
  __shared__ float input_s[BLOCK_DIM * 2];
  const unsigned int tid = threadIdx.x;
  unsigned int pout = 0, pin = 1;

  input_s[tid]          = input[tid];
  input_s[length + tid] = input_s[tid];
  
  __syncthreads();
  for (unsigned int stride = 1; stride < length; stride *= STRIDE_FACTOR) {
    pout = 1 - pout; // swap double buffer indices
    pin  = 1 - pin;
    if (tid >= stride)
	//making reduction
    input_s[pout * length + tid] = input_s[pin * length + tid] + input_s[pin * length + tid - stride];
    else
	//otherwise simply copy the previous value 
    input_s[pout * length + tid] = input_s[pin * length + tid];
    __syncthreads();
  }
  output[tid] = input_s[pout * length + tid];

  return;
}
```

- it's a way to overlap computation and memory transfers 
- In one iteration, a thread reads from one buffer (input) and writes the result to the other buffer (output).
- In the next iteration, the roles of the buffers are switched: the previous output buffer becomes the new input buffer, and vice versa.
- This swapping allows continuous use of data without waiting for other threads to sync up, thereby minimizing synchronization points.

 The double buffering approach minimizes the need for `__syncthreads()` calls, which are typically used to prevent read-after-write hazards. This results in fewer stalls and more continuous computation.


## Stencil

A stencil in computational terms refers to a pattern of computing values on a grid where each point's value is determined based on the values of its neighboring points. This is similar to the process of convolution, often used in image processing, where a matrix (kernel) is slid over data to compute outputs based on the kernel's weighted sum of the input values it covers.

```cpp
void stencil_cpu(const float *in, float *out, const int N) {
  for (int i = 1; i < N - 1; ++i)
    for (int j = 1; j < N - 1; ++j)
      for (int k = 1; k < N - 1; ++k)
        get(out, i, j, k, N) = 
						        c0 * get(in, i, j, k, N) 
						        + c1 * get(in, i, j, k - 1, N) +
                               c2 * get(in, i, j, k + 1, N) 
                               + c3 * get(in, i, j - 1, k, N) 
                               + c4 * get(in, i, j + 1, k, N) 
                               + c5 * get(in, i - 1, j, k, N) +
                               c6 * get(in, i + 1, j, k, N);
}
```

Stencil computations are crucial in fields like computational fluid dynamics, heat conductance, weather forecasting, and electromagnetics. Stencils are often used to approximate derivative values from discrete data points that represents physical quantities.

- **Naive Implementation**: Assign one GPU thread per grid point. Each thread calculates the output for a point without needing to wait for other threads, making the process highly parallel.

```cpp
// Kernel function for GPU
__global__ void stencil_kernel_gpu(const float *__restrict__ in, float *__restrict__ out, const int N) {
  const unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    get(out, i, j, k, N) = 
							c0 * get(in, i, j, k, N) 
						    + c1 * get(in, i, j, k - 1, N)
                           c2 * get(in, i, j, k + 1, N) 
                           + c3 * get(in, i, j - 1, k, N) 
                           + c4 * get(in, i, j + 1, k, N) 
                           + c5 * get(in, i - 1, j, k, N) +
                           c6 * get(in, i + 1, j, k, N);
  }
}
```

Several strategies can improve performance and efficiency:

- **Tiling and Privatization**: Use shared memory on the GPU to reduce the frequency of data transfer between the GPU and its main memory.
- **Coarsening and Slicing**: Adjust the organization of threads and data to better fit the GPU's architecture, enhancing performance without exceeding memory capacities.
- **Register Tiling**: Use GPU registers to store data temporarily, reducing the reliance on shared memory and thus saving space.

#### **Tiling and Privatization**
![](images/Pasted%20image%2020240509123546.png)

Differences:

- each thread still corresponds to a cell in the grid, much like in the naive version
- Before performing computations, this kernel loads necessary data into a shared memory array (`in_s`)

```cpp
// Tiling and Privatization
__global__ void stencil_kernel_tiling_gpu(const float *__restrict__ in, float *__restrict__ out, const int N) {
  const unsigned int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;
  const unsigned int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
  const unsigned int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

  __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
  // Greater than 0 not required since unsigned int are used
  if (i < N && j < N && k < N) {
    in_s[threadIdx.z][threadIdx.y][threadIdx.x] = get(in, i, j, k, N);
  }
  __syncthreads();

  if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
    if (threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM - 1 && threadIdx.y >= 1 &&
        threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
      get(out, i, j, k, N) = c0 * in_s[threadIdx.z][threadIdx.y][threadIdx.x] +
                             c1 * in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1] +
                             c2 * in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1] +
                             c3 * in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x] +
                             c4 * in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x] +
                             c5 * in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x] +
                             c6 * in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x];
    }
  }
}
```

#### Coarsening and Slicing

This optimization strategy addresses the limitations posed by block sizes in GPU computations. By dividing the computation into slices and iterating over them, we enhance the use of the GPU's memory and processing capabilities.

![](images/Pasted%20image%2020240509124333.png)


1. **Thread Coarsening**: Instead of processing the entire 3D matrix at once, each thread block handles one slice of the x-y plane at a time. This iteration happens in the z-direction.
2. **Memory Management**: Only three slices of the input matrix are stored at any time: the previous, current, and next. This reduces memory requirements significantly.
- **Initial Setup**: 3 slices along the z-axis: the current, the previous and next slices into shared memory before starting the computation loop. After computation, update the shared memory to prepare for the next slice.
- **Efficiency**: This method minimizes the amount of shared memory needed and increases the arithmetic intensity, which is closer to the optimal value of 3.25 operations per byte (OP/B).
- **Memory Switching**: Similar to a buffer system, shared memory layers are rotated to continuously reuse and update data without frequent global memory access.

```cpp
__global__ void
    stencil_kernel_coarsening_tiling_gpu(const float *__restrict__ in, float *__restrict__ out, const int N) {
  const unsigned int i_start = blockIdx.z * Z_SLICING;
  const unsigned int j       = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
  const unsigned int k       = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

  __shared__ float in_prev_s[IN_TILE_DIM][IN_TILE_DIM];
  __shared__ float in_curr_s[IN_TILE_DIM][IN_TILE_DIM];
  __shared__ float in_next_s[IN_TILE_DIM][IN_TILE_DIM];
  // Check greater than 0 not needed since index is an unsigned int
  if (i_start - 1 < N && j < N && k < N) {
    in_prev_s[threadIdx.y][threadIdx.x] = get(in, i_start - 1, j, k, N);
  }
  if (i_start < N && j < N && k < N) {
    in_curr_s[threadIdx.y][threadIdx.x] = get(in, i_start, j, k, N);
  }
  for (unsigned int i = i_start; i < i_start + Z_SLICING; ++i) {
    // Check greater than 0 not needed since index is an unsigned int
    if (i + 1 < N && j < N && k < N) {
      in_next_s[threadIdx.y][threadIdx.x] = get(in, i + 1, j, k, N);
    }
    __syncthreads();

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
      if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 &&
          threadIdx.x < IN_TILE_DIM - 1) {
        get(out, i, j, k, N) =
            c0 * in_curr_s[threadIdx.y][threadIdx.x] 
            + c1 * in_curr_s[threadIdx.y][threadIdx.x - 1] 
            + c2 * in_curr_s[threadIdx.y][threadIdx.x + 1] 
            + c3 * in_curr_s[threadIdx.y - 1][threadIdx.x] 
            + c4 * in_curr_s[threadIdx.y + 1][threadIdx.x] 
            + c5 * in_prev_s[threadIdx.y][threadIdx.x] 
            + c6 * in_next_s[threadIdx.y][threadIdx.x];
      }
    }
    __syncthreads();

    in_prev_s[threadIdx.y][threadIdx.x] = in_curr_s[threadIdx.y][threadIdx.x];
    in_curr_s[threadIdx.y][threadIdx.x] = in_next_s[threadIdx.y][threadIdx.x];
  }
}
```


#### Registers optimization

Reducing the use of shared and register memory: only store a single slice of the data grid at any iteration. 
By allocating the next slice's data elements to registers and only transferring them to shared memory when they become the current slice, then moving them back to registers once they become the previous slice. 

```cpp
__global__ void
    stencil_kernel_register_tiling_gpu(const float *__restrict__ in, float *__restrict__ out, const int N) {
  const unsigned int i_start = blockIdx.z * Z_SLICING;
  const unsigned int j       = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
  const unsigned int k       = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

  float in_prev;
  __shared__ float in_curr_s[IN_TILE_DIM][IN_TILE_DIM];
  float in_next;
  // Check greater than not needed since index is an unsigned int
  if (i_start - 1 < N && j < N && k < N) {
    in_prev = get(in, i_start - 1, j, k, N);
  }
  if (i_start < N && j < N && k < N) {
    in_curr_s[threadIdx.y][threadIdx.x] = get(in, i_start, j, k, N);
  }
  
  for (unsigned int i = i_start; i < i_start + Z_SLICING; ++i) {
    // Check greater than not needed since index is an unsigned int
    if (i + 1 < N && j < N && k < N) {
      in_next = get(in, i + 1, j, k, N);
    }
    __syncthreads();
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
      if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 &&
          threadIdx.x < IN_TILE_DIM - 1) {
        get(out, i, j, k, N) =
            c0 * in_curr_s[threadIdx.y][threadIdx.x]
            + c1 * in_curr_s[threadIdx.y][threadIdx.x - 1] 
            + c2 * in_curr_s[threadIdx.y][threadIdx.x + 1]
            + c3 * in_curr_s[threadIdx.y - 1][threadIdx.x]
            + c4 * in_curr_s[threadIdx.y + 1][threadIdx.x]
            + c5 * in_prev + c6 * in_next;
      }
    }
    __syncthreads();
    in_prev                             = in_curr_s[threadIdx.y][threadIdx.x];
    in_curr_s[threadIdx.y][threadIdx.x] = in_next;
  }
}
```

- **Single Slice in Shared Memory**: Only the current slice is maintained in shared memory, reducing the shared memory requirements significantly.
- **Register Usage**: Two additional registers per thread are used to store the previous and next slices' data, enabling quick access and modifications without repeatedly accessing shared memory.
- **Memory Switching**: The stencil computations involve rotating the data between registers and shared memory, reducing the frequency and cost of memory accesses.



Smaller shared memory usage -> scarse resources is not registers but shared memory 




## Merge


![](images/Pasted%20image%2020240509161049.png)



```cpp
// Function for CPU merge computation
void merge_cpu(const int *A, const int dim_A, const int *B, const int dim_B, int *C) {
  int i = 0;
  int j = 0;
  int k = 0;

  while ((i < dim_A) && (j < dim_B)) {
    if (A[i] <= B[j])
      C[k++] = A[i++];
    else
      C[k++] = B[j++];
  }
  if (i == dim_A) {
    while (j < dim_B) { C[k++] = B[j++]; }
  } else {
    while (i < dim_A) { C[k++] = A[i++]; }
  }
}
```

[Merge Sort](../../BSc(italian)/Algoritmi%20e%20Principi%20dell'Informatica/src/10.Sorting.md#Merge%20sort) can be optimized for parallel execution, but it's not just a matter of re-implementing a classical algorithm; it involves strategic data management and computation distribution. Merge sort adheres to a stable sort criterion where identical elements from $A$ precede those from $B$ in list $C$, preserving the input order.

Naive

The basic approach divides the output list C among multiple threads, with each thread responsible for merging a specific section from lists A and B. This division is dynamic, depending on the specific elements of C that each thread is calculating, which can lead to uneven workload distribution among the threads.


![](images/Pasted%20image%2020240509161606.png)





Co rank 


A more sophisticated method involves calculating the "co-rank" of elements, which helps in determining how input elements from A and B are paired to form the output C. The co-rank, defined for an element $C[k]$, is derived from indices $i$ and $j$ such that $i + j = k$. 

$$A[i-1]\le B[j]$$
$$B[j-1]<A[i]$$

The co-rank of an element in the output array \( C \) from a merge operation helps to determine precisely which elements from the two input arrays \( A \) and \( B \) will combine to form that specific output element. Specifically, if you are considering an element \( C[k] \) in the output array, the co-ranks \( i \) and \( j \) from arrays \( A \) and \( B \) respectively, satisfy the condition \( k = i + j \). This means that to compute the value of \( C[k] \), you need to merge elements up to \( i \) from array \( A \) and up to \( j \) from array \( B \).

By utilizing binary search based on co-rank, each thread can quickly determine the exact position (indices \( i \) and \( j \)) in the input arrays \( A \) and \( B \) that it needs to process. 

This method uses a binary search to find the appropriate indices efficiently, optimizing the merge process with a complexity of $O(log N)$.



```cpp
__device__ int co_rank(const int k, const int *__restrict__ A, const int m, const int *__restrict__ B, const int n) {
    int i = min(k, m);   // Start i at the smaller of k or m
    int j = k - i;       // Calculate j based on i
    int i_low = max(0, k - n);  // Lower bound for i
    int j_low = max(0, k - m);  // Lower bound for j
    int delta;
    bool active = true;

    while (active) {
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            // If A[i-1] is greater than B[j], adjust i and j
            delta = max(1, (i - i_low) / 2);  // Ensure delta is at least 1 and calculate half the interval
            i -= delta;
            j += delta;
        } else if (j > 0 && i < m && B[j - 1] >= A[i]) {
            // If B[j-1] is greater than or equal to A[i], adjust i and j
            delta = max(1, (j - j_low) / 2);  // Ensure delta is at least 1 and calculate half the interval
            i += delta;
            j -= delta;
        } else {
            // If neither condition is met, we've found the correct i
            active = false;
        }
    }

    return i;  // Return the co-rank i
}
```

With clear co-rank values, each thread or thread block can independently compute a segment of the output array without needing to wait for other threads to complete their tasks. 

In practical scenarios, the input arrays $A$ and $B$  might not be of equal size, which can complicate the parallel processing of data. Co-rank helps manage these irregularities by providing a systematic way to allocate elements of $A$ and $B$ to threads regardless of their lengths. 


Further optimizations include the use of shared memory and registers.

#### Tiling variant

Each iteration of the merge operation involves loading specific segments ("tiles") of lists A and B into shared memory. In this strategy, during the first iteration, segments from both lists are uploaded into shared memory. All threads then evaluate which parts of these segments they need to merge based on their specific task within the overall merge operation. 

![](images/Pasted%20image%2020240509172739.png)

In subsequent iterations, new segments of A and B are loaded into shared memory. It’s crucial to note that the entire shared memory may not always be filled if the segments from A and B do not align perfectly, potentially leading to inefficiencies and the introduction of bugs.


#### Circular buffering

One proposed enhancement involves using a circular buffering method to optimize the utilization of shared memory across iterations. Instead of reloading data that has already been brought into shared memory, the algorithm can maintain a dynamic index that tracks where new data should be written and read within the shared memory. This approach reduces the redundancy of memory accesses and maximizes the use of already-loaded data.




- If `A` and `B` are the input arrays, and `k` is the index in the merged array `C`:
    - `co_rank(k, A, dim_A, B, dim_B)` finds the position in `A` such that the sum of elements from `A` and `B` up to this position is equal to `k`.

Let's visualize the co-rank for `k = 2` in our example:

- `A = [1, 3, 5, 7]`
- `B = [2, 4, 6, 8]`

We want to find how many elements from `A` and `B` make up the first 2 elements in `C`:

- The co-rank splits the merged array such that the first `k` elements are from `A` and `B`.

In this case:

- `co_rank(2, A, 4, B, 4)` would find that:
    - 1 element from `A` (1) and 1 element from `B` (2) make up the first 2 elements in `C`.



After loading the data into the shared memory buffers, we need to update the starting index for the next iteration of the circular buffer:


```cpp
Copy code
A_S_start = (A_S_start + A_S_consumed) % TILE_SIZE;
B_S_start = (B_S_start + B_S_consumed) % TILE_SIZE;
```

- `A_S_start`: The starting index in the circular buffer A_s.
- `A_S_consumed`: The number of elements consumed (loaded) in the current iteration for `A`.


###  Sparse Matrix Formats

Sparse matrix computation is crucial in scientific, engineering, and financial modeling problems, where the majority of matrix elements are zeros. Storing and processing these zeros wastes memory and bandwidth. To address this, various sparse matrix formats have been developed to compactly represent the non-zero elements, influencing GPU performance. 
These formats include:
- COO (Coordinate Format)
- ELL (ELLpack Format)
- CSR (Compressed Sparse Row).


Sparse matrices often arise in applications like solving systems of equations, where each row represents one equation, and in modeling networks where the connections are sparse. 

With this parameters:

- **M**: Number of rows in the matrix
- **N**: Number of columns in the matrix
- **K**: Number of nonzero entries in the densest row
	- **S**: Sparsity level `[0-1]`, `1` being fully-dense

| Format                      | Storage Requirements |
| --------------------------- | -------------------- |
| Dense                       | $MN$                 |
| Compressed Sparse Row (CSR) | $2MNS + M + 1$       |
| ELL                         | $2MK$                |
| Coordinate (COO)            | $3MNS$               |
| Hybrid ELL / COO (HYB)      | $2MK << 3MNS$        |

#### CSR

The CSR format stores only the non-zero values and their corresponding column indexes, which impacts memory usage, data modification efficiency, and access patterns, affecting memory bandwidth peak performance and workload balance among threads.


![](images/Pasted%20image%2020240518183331.png)


Compressed Sparse Row (CSR) is space-efficient but less flexible. It stores each row as a sparse vector, requiring additional storage to locate the start of each row. The space requirements are defined as $2MNS + M + 1$, and CSR is space-saving if $S < \left(1 - \frac{1}{N}\right) / 2$. 

Assigning a thread to each row for parallel SpMV (Sparse Matrix-Vector multiplication) ensures each thread handles a distinct output value, but consecutive threads accessing distant memory locations result in inefficient memory bandwidth utilization and potential control flow divergence.

ELL format
![](images/Pasted%20image%2020240518184106.png)

originating from ELLPACK, relies on the maximum number of non-zero elements per row and **padding** elements. It's more flexible than CSR since non-zero elements can be added by replacing padding elements. ELL uses memory bandwidth efficiently due to coalesced memory access but can lead to load imbalance. The space requirements for ELL are $2MK$, and it saves space if $K < \frac{N}{2}$.

### COO

COO format stores both column and row indexes for every non-zero element, leading to higher space requirements but offering flexibility in adding non-zero elements. 


Parallel SpMV in COO assigns each thread to compute a non-zero element, balancing the workload but requiring atomic operations for synchronization. The space requirements for COO are $3MNS$, and it saves space if $S < \frac{1}{3}$.

```cpp
struct SparseMatrixCOO {
  int* row_indices;
  int* column_indices;
  float* values;
  int num_elements;
}
```

![](images/Pasted%20image%2020240518184710.png)


```cpp
void SpMV_COO(const SparseMatrixCOO* A, const float* x, float* y) {
  for (int element = 0; element < A->num_elements; ++element) {
    const int column = A->column_indices[element];
    const int row    = A->row_indices[element];
    y[row] += A->values[element] * x[column];
  }
}
```

And in GPU:

```cpp
__global__ void SpMV_COO_gpu(
const int* __restrict__ row_indices,
const int* __restrict__ column_indices,
const float* __restrict__ values,
const int num_elements,
const float* x,
float* y) {

for (int element = threadIdx.x + blockIdx.x * blockDim.x; element < num_elements;element += blockDim.x * gridDim.x) {
    const int column = column_indices[element];
    const int row    = row_indices[element];
    atomicAdd(&y[row], values[element] * x[column]);
  }
  
}
```


### Hybrid ELL/COO

![](images/Pasted%20image%2020240518185804.png)


A hybrid ELL/COO format addresses ELL's inefficiency in handling rows with many non-zeros by placing these non-zeros in a COO format, leading to a more efficient ELL representation for the remaining elements. This approach enhances space efficiency and flexibility, making it beneficial for iterative solvers despite the overhead of using two different formats.
The code is basically the same but just combines the two previous methods. 

### Extra 

Other formats include:

- **Jagged Diagonal Storage (JDS):**
  - Groups similarly dense rows into partitions.
  - Represents partitions independently using either CSR or ELL.
  - Sorts rows by density, avoiding padding and improving space efficiency.
  - Allows coalesced memory access but lacks flexibility in adding new elements.
- **Diagonal (DIA):**
  - Stores only a sparse set of dense diagonal vectors.
  - For each diagonal, the offset from the main diagonal is stored.
- **Packet (PKT):**
  - Reorders rows and columns to concentrate nonzeros into roughly diagonal submatrices.
  - Improves cache performance as nearby rows access nearby x elements.
- **Dictionary of Keys (DOK):**
  - Stores the matrix as a map from (row, column) index pairs to values.
  - Useful for building or querying a sparse matrix, but iteration is slow.
- **Compressed Sparse Column (CSC):**
  - Similar to CSR but stores a dense set of sparse column vectors.
  - Useful when column sparsity is much more regular than row sparsity.
- **Blocked CSR:**
  - Divides the matrix into blocks stored using CSR with the indices of the upper left corner.
  - Useful for block-sparse matrices.


In general, the FLOPS ratings achieved by both CPUs and GPUs are much lower for sparse matrix computation than for dense matrix computation. In SpMV (Sparse Matrix-Vector multiplication) computation, there is no data reuse in the sparse matrix, which results in a low operational intensity. The operational intensity (OP/B) is essentially 0.25, significantly limiting the achievable FLOPS rate to a small fraction of the peak performance compared to dense matrix computations.
