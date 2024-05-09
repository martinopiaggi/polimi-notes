
# Task level parallelism streams, events, dynamic parallelism 

In the first classes, we commented that CUDA supports two different types of parallelism: data parallelism and task parallelism. 

During the first part of this course, we focused on data parallelism because the GPU is designed to exploit it through its parallel architecture. We dissected a single function to parallelize the execution on each data chunk or item. We also discussed mechanisms for parallelizing kernels. In our implementations, there was still sequential behavior due to data transfer and kernel execution being executed sequentially. However, the GPU works in parallel with the CPU, allowing for coarse-grained concurrency. By executing different functions on the CPU and GPU simultaneously, multiple kernels can be executed concurrently. The goal is to understand how to enable task-level parallelism, parallelize CPU and GPU execution, and data transfers.

To summarize, the programmer is responsible for enabling task-level parallelism by queuing commands using a software queue called a stream. Commands are pushed and popped using a first-in-first-out policy and forwarded to the GPU. On the GPU side, everything is managed directly in hardware, with commands executed sequentially. Data transfers are blocking, meaning the thread on the host waits until the entire transmission is executed, while kernel launches allow the host to continue the execution. The workflow on the GPU is fully sequential."

## Data Parallelism

- **Focus of the Course**: The GPU architecture is inherently designed to excel at data parallelism. This involves dissecting a function and parallelizing its execution across multiple pieces of data simultaneously.
- **Example**: Breaking down image processing tasks into smaller chunks where each GPU thread processes one pixel independently.

## Task Parallelism

- **Coarse-grained Concurrency**: CUDA also supports task parallelism, where different tasks are performed simultaneously by coordinating CPU and GPU operations. This allows for more complex orchestrations of computations and data management.

### Streams and Concurrency

- **Streams**: CUDA introduces the concept of streams for task-level parallelism, which involves queuing commands in a software-managed queue. Commands in a stream are executed sequentially on the GPU, but multiple streams can operate in parallel, providing task concurrency.
- **Execution Model**:
  - **Data Transfers**: Operations like memory transfers are typically blocking, where the CPU waits for the completion before proceeding.
  - **Kernel Execution**: Launching kernels can be non-blocking, allowing the CPU to continue executing other tasks.

### Practical Implications
- **Parallel Execution**: By using streams, programmers can effectively overlap data transfers and kernel executions, optimizing the utilization of both CPU and GPU resources.
- **Responsibility**: It's up to the programmer to design the application to take full advantage of task-level parallelism by carefully scheduling operations and managing dependencies.


- CUDA GPUs, especially from the Fermi architecture onwards, support concurrent execution of tasks using multiple hardware queues.
- These queues allow for independent kernels or data transfers to be processed in parallel, enhancing overall throughput and efficiency.




Understanding the distinction between blocking and non-blocking commands in CUDA is crucial for optimizing the performance of GPU-accelerated applications. Here, we explore both types of commands and their implications on program execution.

## Blocking Commands

- **Definition**: Blocking commands halt the execution of the host program until the GPU operation completes.
- **Example**: `cudaMemcpy`, which synchronously copies data between the host and the GPU.
```cuda
  cudaMemcpy(devPtr, hostPtr, count, cudaMemcpyHostToDevice);
```
## Non-Blocking Commands

- **Definition**: Non-blocking commands allow the host program to continue execution asynchronously while the GPU operation is still running.
- **Examples**:
    - **Kernel Launch**: By default, kernel launches are non-blocking.
        
        cudaCopy code
        
        ```cpp
        kernel<<<numBlocks, blockSize>>>(arguments);
        ```
        
        - The host program can proceed immediately after the kernel is dispatched to the GPU.
    - **Asynchronous Data Transfer**: Using `cudaMemcpyAsync` with streams.
        cudaCopy code
        
        ```cuda
        Stream_t stream; cudaStreamCreate(&stream); // Asynchronously copy data to the GPU cudaMemcpyAsync(devPtr, hostPtr, count, cudaMemcpyHostToDevice, stream); // Launch kernel with stream kernel<<<numBlocks, blockSize, 0, stream>>>(arguments); // Copy results back to host asynchronously cudaMemcpyAsync(hostPtr, devPtr, count, cudaMemcpyDeviceToHost, stream); cudaStreamDestroy(stream);
        ```
        
        - These commands do not block the host, which can perform other tasks while the GPU is processing.

- CUDA provides software layers and commands to exploit these architectural features:
    ```cuda
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaMemcpyAsync(devPtr1, hostPtr1, size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(devPtr2, hostPtr2, size, cudaMemcpyHostToDevice, stream2);
    kernel1<<<blocks, threads, 0, stream1>>>(arguments1);
    kernel2<<<blocks, threads, 0, stream2>>>(arguments2);
    ```
    - **Multiple Streams**: By using multiple streams, tasks can be queued independently, allowing them to be executed on the GPU concurrently.
    - **Asynchronous Commands**: Non-blocking memory transfers (`cudaMemcpyAsync`) allow the CPU to continue other tasks while the data transfer is underway.

## Exploiting Task-Level Parallelism

- **Worker Distributor Role**:
    - In CUDA's execution model, a worker distributor acts like a scheduler, managing tasks across different hardware queues.
    - Each queue executes commands sequentially, but commands from different queues can be processed concurrently, enabling true task-level parallelism.



Understanding Pinned Memory and Asynchronous Data Transfer in CUDA

Since the virtual memory mapping and the rotation of pages by the OS can take to undefeined behaviour. CUDA runtime solve this problem using pinned (locked) pages. 

We can force a pinned page directly from the code. 

Pinned memory plays a crucial role in optimizing data transfers between the host and GPU, offering a significant performance boost by avoiding unnecessary copies and enabling asynchronous operations. When memory is allocated as "pinned" or "page-locked," it cannot be swapped out by the operating system, ensuring the memory address remains constant and accessible for DMA (Direct Memory Access) operations.

Pinned Memory Benefits:
- **Increased Data Transfer Performance**: Direct access by the DMA engine eliminates the need for copying data to a temporary pageable buffer before transferring to the GPU.
- **Enabling Asynchronous Data Transfer**: Asynchronous transfers allow the GPU to perform data transfer concurrently with other tasks, thereby maximizing the utilization of hardware resources.

However, using pinned memory has its drawbacks:
- **Resource Constraints**: Pinned memory can limit the flexibility of the system’s virtual memory management by restricting the OS from dynamically managing the memory space.
- **Potential Impact on Overall System Performance**: Overuse of pinned memory might degrade the performance of other applications or the system as a whole due to reduced memory flexibility.

Implementing Pinned Memory and Asynchronous Transfers:

The process involves replacing standard memory allocation methods with those specifically designed for pinned memory:
- **Memory Allocation**:
  ```cpp
  cudaMallocHost(&ptr, size);  // Allocates pinned memory on the host
  ```
  This replaces typical `malloc` or automatic/static array declarations with `cudaMallocHost`, which directly communicates with the OS to allocate memory in a pinned manner.

- **Asynchronous Data Transfer**:
  ```cpp
  cudaMemcpyAsync(destination, source, size, cudaMemcpyHostToDevice, stream);
  ```
  `cudaMemcpyAsync` is the asynchronous counterpart of `cudaMemcpy`, pushing the data transfer command into a CUDA stream. This function allows the host thread to continue execution without waiting for the data transfer to complete.

- **Synchronization Point**:
  ```cpp
  cudaDeviceSynchronize();
  ```
  Synchronization after asynchronous operations is crucial to ensure that all operations complete before the program proceeds, as data integrity might be compromised if the host accesses data prematurely.




**Releasing Host Resources and Managing CUDA Streams for Enhanced Parallel Execution**

Releasing resources in CUDA involves using `cudaFreeHost`, which is crucial after operations are complete to avoid unnecessary memory consumption on the host. This function is especially significant when dealing with pinned memory to ensure resources are freed properly, maintaining system health and performance.

Impact on System Performance

Using pinned memory has notable advantages, such as faster data transfer rates due to the elimination of a memory copy step. However, it restricts the operating system's ability to manage memory dynamically, potentially affecting the performance of other applications or the overall system. Thus, while pinned memory enhances single memory transmission speed, it imposes constraints on the operating system's flexibility in memory management.



After the GPU completes its tasks, a synchronization point is essential. Using `cudaDeviceSynchronize` ensures that all commands submitted to the GPU finish before the CPU attempts to use the results. This synchronization acts as a barrier, preventing premature access to data that the GPU is still processing.

**Enabling Higher Levels of Parallelism**

To truly leverage the power of CUDA for parallel processing, using multiple streams (queues) allows for concurrent execution of commands on the GPU. Each stream can independently queue operations like memory transfers or kernel executions, which the GPU can then process as resources become available. This method significantly enhances the capacity for parallel task execution.

**Practical Application and Stream Management**

Here's a practical application of managing CUDA streams to enhance parallelism:

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);
// Asynchronously copy data to the GPU
cudaMemcpyAsync(devPtr, hostPtr, count, cudaMemcpyHostToDevice, stream);
// Launch kernel with stream
kernel<<<numBlocks, blockSize, 0, stream>>>(arguments);
// Copy results back to host asynchronously
cudaMemcpyAsync(hostPtr, devPtr, count, cudaMemcpyDeviceToHost, stream);
cudaStreamDestroy(stream);
```

In this example, different tasks such as data transfers and kernel execution are handled in a single stream, allowing for non-blocking operations that let the CPU and GPU work more efficiently in parallel. This setup minimizes the idle time on the CPU and enables it to perform other tasks while the GPU processes the data.

For applications requiring even greater concurrency, employing multiple streams can parallelize the execution further. This approach allows overlapping of multiple operations, such as different kernel executions or simultaneous data transfers, which can occur on separate streams. Here’s how you can manage multiple streams:

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
// Operations on stream1
cudaMemcpyAsync(devPtr1, hostPtr1, size, cudaMemcpyHostToDevice, stream1);
kernel<<<blocks, threads, 0, stream1>>>(args1);
// Operations on stream2
cudaMemcpyAsync(devPtr2, hostPtr2, size, cudaMemcpyHostToDevice, stream2);
kernel<<<blocks, threads, 0, stream2>>>(args2);
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```
Using multiple streams enhances the application's ability to perform multiple tasks concurrently, significantly improving performance and efficiency.

CUDA provides two types of streams: 

- the default stream
- non-default streams. 

The default stream, or NULL stream, is automatically handled and requires no explicit intervention from the programmer. However, to achieve higher levels of parallelism and control, non-default streams must be explicitly declared and managed.

**Using Non-Default Streams**
- To utilize non-default streams, you first declare and initialize them using CUDA API calls:
  ```cpp
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  ```
- These streams allow for the parallel execution of tasks such as kernel execution and data transfers, enhancing the GPU's ability to perform multiple operations concurrently.

**Memory Considerations**
- It is important to allocate pinned memory on the CPU side when using non-default streams. Pinned memory is crucial for asynchronous data transfers, ensuring that memory is not paged by the OS and is always available for quick GPU access:
  ```cpp
  int *hostArray;
  cudaMallocHost(&hostArray, size);  // Allocate pinned memory
  ```

**Executing Commands in Streams**
- When executing commands such as kernel launches or memory copies, specify the target stream as the last parameter:
  ```cpp
  cudaMemcpyAsync(deviceArray, hostArray, size, cudaMemcpyHostToDevice, stream1);
  kernel<<<gridSize, blockSize, 0, stream1>>>(args);
  ```
- This ensures that these commands are queued in the specified stream and can execute concurrently with other streams.

**Synchronizing Streams**
- Synchronization is essential to ensure that all commands in a stream complete before the CPU proceeds with operations that depend on the results:
  ```cpp
  cudaStreamSynchronize(stream1);
  ```
- This places a barrier, ensuring the CPU only continues once all commands in `stream1` are complete.

**Behavior and Execution**
- Commands within the same stream execute in a FIFO (First In, First Out) order without overlapping. However, commands in different streams can execute out of order and concurrently, depending on resource availability.

**Handling Multiple Streams**
- Multiple streams are used to parallelize tasks further:
  ```cpp
  cudaStream_t stream2;
  cudaStreamCreate(&stream2);
  // Example of overlapping different streams
  cudaMemcpyAsync(deviceArray1, hostArray1, size1, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(deviceArray2, hostArray2, size2, cudaMemcpyHostToDevice, stream2);
  ```
- This setup allows the GPU to switch between tasks in different streams, potentially running them in parallel if resources allow.

**Special Rules for Default Stream**
- Operations in the default stream are executed only after all operations in other streams are complete, and vice versa. This ensures a level of synchronization without explicit barriers but can limit concurrency.

**Advanced Stream Management**
- To avoid blocking behavior typical of the default stream, use the `cudaStreamCreateWithFlags` function to create non-blocking streams:
  ```cpp
  cudaStream_t stream3;
  cudaStreamCreateWithFlags(&stream3, cudaStreamNonBlocking);
  ```

**Practical Example of Stream Usage**
- Implementing three different kernel operations in separate streams demonstrates potential parallel execution:
  ```cpp
  cudaStream_t stream1, stream2, stream3;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);
  kernel1<<<grid, block, 0, stream1>>>(...);
  kernel2<<<grid, block, 0, stream2>>>(...);
  kernel3<<<grid, block, 0, stream3>>>(...);
  ```

In summary, effective management of CUDA streams facilitates the optimization of parallel executions, allowing the GPU to handle multiple tasks simultaneously. This approach is key to maximizing performance in applications requiring high levels of computational power.

![](images/Pasted%20image%2020240422210942.png)


HW mechanisms to support multiple streams

With the evolution of GPU architectures from Fermi to Kepler and beyond, NVIDIA introduced the ability to handle multiple hardware queues, thus enabling more sophisticated management of parallel tasks through the use of multiple streams.

Modern GPUs typically support 32 hardware streams. If the number of declared streams exceeds this limit, multiple software streams may map to the same hardware queue, potentially leading to contention and reduced performance.



**Stream Management**:
- In CUDA, a stream is a sequence of commands (like kernel executions or memory operations) that execute in order. CUDA allows the creation of multiple streams for handling different tasks concurrently.
- When multiple streams are used, the CUDA driver manages the execution order across different hardware queues. This was limited to a single queue in older Fermi GPUs, creating potential false dependencies among tasks.

**Example of Stream Utilization**:

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);  // Create first non-default stream
cudaStreamCreate(&stream2);  // Create second non-default stream

// Operations on stream1
cudaMemcpyAsync(devPtr1, hostPtr1, size, cudaMemcpyHostToDevice, stream1);
kernel<<<numBlocks, blockSize, 0, stream1>>>(arguments1);

// Operations on stream2
cudaMemcpyAsync(devPtr2, hostPtr2, size, cudaMemcpyHostToDevice, stream2);
kernel<<<numBlocks, blockSize, 0, stream2>>>(arguments2);

cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

### Synchronization Techniques

**Host and Device Synchronization**:
- Synchronization is necessary to ensure that data dependencies are respected among various operations. CUDA provides several mechanisms to synchronize operations both within a single stream and across multiple streams.


Implicit Synchronization in CUDA Function Calls

In CUDA programming, synchronization between the host (CPU) and the device (GPU) is critical for ensuring that data dependencies and execution orders are correctly maintained. Certain CUDA function calls inherently synchronize the host with the device, meaning that the host execution will block until the specified CUDA operations are completed. This behavior ensures that data being processed by these operations is fully updated and consistent before the host proceeds with further execution.

#### Functions that Implicitly Synchronize Host and Device

1. **Memory Allocation Functions**:
   - **`cudaMallocHost()`**:
     - Allocates host memory that is page-locked and accessible to the device. This function blocks the host until the memory is fully allocated to prevent any access to uninitialized memory.
   - **`cudaHostAlloc()`**:
     - Similar to `cudaMallocHost()`, it allocates page-locked memory but with more control over the properties (like whether the memory is write-combined).

2. **Device Memory Allocation**:
   - **`cudaMalloc()`**:
     - Allocates memory on the device. The host is blocked until the completion of this operation to ensure that any subsequent operations related to this memory (like memory transfers or kernel executions using this memory) are valid.

3. **Synchronous Memory Copy Functions**:
   - **`cudaMemcpy*()`** — non-Async:
     - Copies data between host and device or between different memory areas on the device. When using non-async versions of these functions, the host will wait until the entire data transfer is completed before proceeding.

4. **Memory Setting Functions**:
   - **`cudaMemset*()`** — non-Async:
     - Sets memory with a constant byte value. The non-async version blocks the host until the device has completed setting the specified memory, ensuring that any usage of this memory reflects the new values.

5. **Device Configuration**:
   - **`cudaDeviceSetCacheConfig()`**:
     - Sets the cache configuration for the device. This function blocks the host until the GPU has completed the configuration change, preventing any kernels that might rely on this configuration from launching prematurely.

#### Implications of Implicit Synchronization

While implicit synchronization can be helpful in managing data consistency and simplifying program logic by ensuring that certain operations are complete before proceeding, it can also lead to reduced performance if not managed carefully. Blocking the host can prevent it from performing other tasks that could otherwise overlap with GPU operations, thus reducing the overall efficiency of the application.

To minimize the performance impact of these synchronization points, consider the following strategies:
- **Asynchronous Operations**: Where possible, use asynchronous versions of memory operations (`cudaMemcpyAsync()`, `cudaMemsetAsync()`) along with CUDA streams to allow overlap of memory transfers with computation, both on the host and device.
- **Stream Synchronization**: Use `cudaStreamSynchronize()` to synchronize only specific streams rather than blocking the entire host, allowing other operations to continue concurrently.

By understanding and effectively managing these implicit synchronization points in CUDA




**Using Events for Synchronization**:
- CUDA events are markers that can be placed in streams to record execution status. They are particularly useful for profiling and for coordinating between the host and device:

```cpp
cudaEvent_t event;
cudaEventCreate(&event);
cudaMemcpyAsync(devPtr, hostPtr, size, cudaMemcpyHostToDevice, stream1);
cudaEventRecord(event, stream1);
cudaStreamWaitEvent(stream2, event, 0); // Make stream2 wait for event
cudaEventDestroy(event);
```

**Global and Fine-Grained Synchronization**:
- `cudaDeviceSynchronize()`: Waits for the completion of all device activities and is considered a global barrier.
- `cudaStreamSynchronize(stream1)`: Synchronizes specific streams, allowing other streams to continue execution, providing more control and potentially reducing idle time.

**Using Non-Blocking Streams**:
- Non-blocking streams (`cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)`) allow operations within the stream to proceed without having to wait for operations in the default stream to complete, further enhancing concurrency.




## Dynamic parallelism 

Dynamic parallelism allows any thread within a kernel to potentially submit another kernel call. 
Highlights:

- **Recursive Kernel Launches**: Kernels can invoke other instances of themselves to tackle simpler problems.
- **Adaptive Thread Management**: Useful for adjusting the number of threads for data sets with unpredictable sizes, like graphs or sparse matrices, which may vary during processing.
- **Reduced Host-Device Communication**: Eliminates the need for constant signaling between the host and device when additional tasks are found, allowing threads to directly launch new grids and bypassing the usual host intervention.

![](images/Pasted%20image%2020240508174320.png)

Dynamic approach where the kernel starts with the coarse-grain grid configuration and then eventually spawns multiple child grids. 

**Unbalanced Matrix Example**: Considers a matrix where each row has a different number of columns, discovered during runtime.

```cpp
__global__ void kernel(unsigned int* start, unsigned int* end, float* someData, float* moreData) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    doSomeWork(someData[i]);
    for (unsigned int j = start[i]; j < end[i]; ++j) {
        doMoreWork(moreData[j]);
    }
}
```

Threads experience branch divergence due to the variable row lengths in `moreData`.
- **Solution Exploration**:
  - **Coarse-Grained Approach**: Threads sequentially process each row but may idle if they have no data to process, leading to inefficiencies.
  - **Fine-Grained Approach**: Involves calculating maximum vector sizes and using a 2D grid, but results in high resource use by inactive threads.
  - **Dynamic Solution**:
    - Utilizes a two-step kernel approach where a parent kernel first processes `someData`.
    - Based on the data processed, it dynamically launches child kernels to handle `moreData`, adjusting grid sizes to the workload.
    - This method aims to optimize thread activity and resource allocation by dynamically fitting the kernel execution to the data requirements. 


- **Kernel Capabilities**: CUDA kernels can:
  - Launch other kernels
  - Allocate global memory
  - Synchronize with child grids
  - Create streams

  However, the range of functions callable from within a device is more limited compared to those callable from the host.

- **Asynchronous Child Kernel Launch**: 
  - Each thread can independently launch a child kernel, which is asynchronous.
  - To prevent launching a child grid per thread, only a specific thread (e.g., the first thread in a block or grid) should be programmed to launch the kernel to avoid creating excessive child grids.

- **Execution and Synchronization**: 
  - There's no predefined execution order between parent and child threads; the order is determined dynamically.
  - Parent and child threads synchronize only at specific points—during launch and termination of the child grid.
  - An implicit barrier in the parent thread ensures that child grids complete before the parent continues, securing a proper sequence of operations.

- **Grid Nesting and Synchronization**: 
  - Child grids are nested within parent grids and must complete before the parent thread resumes, implementing a last-in-first-out (LIFO) execution pattern.
  - Synchronization is critical and is managed with `CUDA device synchronize`, which ensures that all operations within the thread (and potentially across blocks) are complete before proceeding.

- **Concurrency Concerns**: 
  - The actual scheduling of child grids relative to parent threads is not guaranteed to be immediate and may occur later in the kernel's execution.
  - If no explicit synchronization function is used within the parent kernel, the scheduling of child kernels follows the execution block, ensuring completion before the parent block proceeds.

- **Resource and Context Management**: 
  - Potential issues arise when resources are insufficient to accommodate a newly launched child grid, leading to context swapping or delays.
  - The management of these resources and the execution context is complex, especially when considering multi-core or multi-processor environments.


- **Context Switching Costs**: Context switching on the GPU, referred to as "Application-level context," is resource-intensive as it involves storing the GPU's current context in memory to allow another application to run.
- **Memory Data Visibility**: Child grids can access the parent grid's mobile, texture, and constant memory. However, simultaneous operations on the same data structure by both grids are not recommended due to the lack of guaranteed execution order, leading to weak consistency.
- **Stream Management**:
  - By default, launching a kernel without specifying a stream uses the null stream, leading to serialized execution of child grids launched by multiple threads in the same block.
  - To enable parallel execution, kernels should be launched in separate streams created with the CUDA stream non-blocking flag. Proper synchronization is necessary before stream destruction.
- **Dynamic Parallelism Costs**:
  - Dynamic parallelism allows a depth of up to 24 levels theoretically, but practical limits due to GPU resource constraints are much lower.
  - The number of pending kernels is capped at 2048, and virtualization involves moving data to global memory, which can degrade performance.








