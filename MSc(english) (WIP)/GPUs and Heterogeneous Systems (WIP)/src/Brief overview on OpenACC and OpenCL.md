# Brief overview of  OpenACC 

## OpenACC 

OpenACC is a directive-based language used to accelerate code on various architectures, not just GPUs. It simplifies the process compared to CUDA, especially in terms of restructuring code and data management through concepts like unified memory. The goal is to make code acceleration more efficient and less time-consuming:

- **compiler-based** acceleration: annotations to guide the compiler in restructuring the code for acceleration. 
- **high portability**: the compiler can target different architectures like GPUs or multicore CPUs just by recompiling. This allows quick acceleration even for non-skilled programmers, reducing time to market. 

In OpenACC, the parallelism model involves:

- **Gangs**: Corresponding to CUDA blocks, fully independent execution units.
- **Workers**: Each elaborates a vector of work, can synchronize within a gang. **Workers as CUDA Threads**: Each worker executes the same instructions on multiple data.
- **Vectors**: Execute the same instruction on multiple data, similar to SIMD execution. Each vector element is considered a CUDA thread.

Compiler options can target both GPUs and CPUs without altering the source code. The compiler identifies loop parallelism and transforms sequential code into parallel code efficiently.
```c
#pragma acc kernels
for(int i = 0; i < N; i++)
    C[i] = A[i] + B[i];
```
The kernel directive lets the compiler decide on parallelization, while the parallel directive requires the programmer to specify it for each loop.
```c
#pragma acc parallel
{
    #pragma acc loop
    for(int i = 0; i < N; i++)
        C[i] = A[i] + B[i];
}
```

The OpenACC compiler can improve performance by collapsing nested loops and utilizing kernel and parallel directives. 

```c
#pragma acc parallel loop collapse(2)
for(int i = 0; i < N; i++)
    for(int j = 0; j < N; j++)
        C[i*N+j] = A[i*N+j] + B[i*N+j];
```

Without `collapse(2)` Parallelization might be limited by the outer loop while with it The two loops are collapsed into a single loop and parallelization can be applied across N * N iterations.


**`seq` Directive**: Executes a loop sequentially within a parallel region. Applying a sort of **coarsening** strategy.

```c
#pragma acc parallel loop
for (int i = 0; i < N; i++) {
    #pragma acc loop seq
    for (int j = 0; j < N; j++) {
        C[i*N + j] = A[i*N + j] + B[i*N + j];
    }
}
```

**`reduction` Directive**: Performs parallel reductions with supported operators `+`, `*`, `max`, `min`, `&`, `|`, `&&`, `||`.

```c
int sum = 0;
#pragma acc parallel loop reduction(+:sum)
for (int i = 0; i < N; i++) {
    sum += A[i];
}
```

**`gang(x)` clause**: divides a loop into `x` independent gangs for parallel execution. Each gang can run independently on different processing units. `worker` os then optionally used to specify how to parallelize loops. 

```c
#pragma acc parallel loop gang(16)
for (int i = 0; i < N; i++) {
    #pragma acc loop worker
    for (int j = 0; j < N; j++) {
        C[i*N + j] = A[i*N + j] + B[i*N + j];
    }
}
```

Proper handling of private variables with clauses like `private` is essential to avoid issues of global access.

```c
#pragma acc parallel loop private(swap)
for(int i = 0; i < N; i++){
    int swap = A[i];
    A[i] = B[i];
    B[i] = swap;
}
```

Different levels of parallelism can be specified for each loop using directives like `gang`.
Commands like `copyin`, `copyout`, and `create` handle data transfer tasks: 

- **`copyin`**: Data are copied from the host to the device at the beginning of the parallel region, and memory is freed at the end of the region.
- **`copyout`**: Device memory is allocated at the beginning of the parallel region and copied to the host memory at the end of the region.
- **`copy`**: Combines `copyin` and `copyout` behaviors.
- **`create`**: Allocates device memory.

The compiler then decides how and when to move data between the host and the device. 

```c
#pragma acc data copyin(A[0:N], B[0:N]) copyout(C[0:N])
{
    #pragma acc kernels
    for(int i = 0; i < N; i++)
        C[i] = A[i] + B[i];
}
```

### Example of Jacobi iterative method 

Jacobi iterative method that solves the Laplace equation to compute the heating transfer on a 2D surface.

```c
#pragma acc data create(Anew[0:n][0:m]) copy(A[0:n][0:m])
while (err > tol && iter < iter_max) {
    err = 0.0;
    #pragma acc parallel loop reduction(max:err) collapse(2)
    for(int j = 1; j < n-1; j++) {
        for(int i = 1; i < m-1; i++) {
        //temperature of each single point is computed as 
        // linear comb of 4 neighbours 
            Anew[j][i] = 0.25 * (A[j][i+1] 
            + A[j][i-1] 
            + A[j-1][i] 
            + A[j+1][i]);
            err = max(err, abs(Anew[j][i] - A[j][i]));
        }
    }
    #pragma acc parallel loop collapse(2)
    for(int j = 1; j < n-1; j++) {
        for(int i = 1; i < m-1; i++) {
            A[j][i] = Anew[j][i];
        }
    }
    iter++;
}
```


### Parallelization Considerations for Loops

- **While loop**: Has data dependencies among iterations and cannot be parallelized.
- **Nested for loops**: Iterations are independent and can be parallelized so `collapse(2)`
- **Error computation**: `err` is computed using a reduction clause.
- **Data transfer**:
  - `A`: Transferred to the device at the beginning and back to the host at the end. `copy(A[0:n][0:m])`
  - `Anew`: Used only on the device, so `create(Anew[0:n][0:m])` 


## OpenCL 

OpenCL acts as a runtime that enables the programming of diverse computing resources using a single language, addressing the challenge of managing multiple accelerators and programming languages.

CUDA is designed by NVIDIA and it's tailored to specific accelerators, leading to the development of OpenCL as a comprehensive solution. 

OpenCL has evolved to support different types of accelerators beyond just GPUs, including those found in desktops, HPC systems, service machines, mobile devices, and even FPGAs. The goal is to optimize performance while considering energy consumption, especially in mobile markets where efficiency and trade-offs between compute units like CPUs and GPUs are crucial.

The development of OpenCL was spearheaded by Apple and is now managed by the Khronos Group, a consortium of hardware and software companies. 

Supported Parallelism Models: 

- **Single-Instruction-Multiple-Data (SIMD)**
	- The kernel is composed of sequential instructions.
	- The instruction stream is executed in lock-step on all involved processing elements.
	- Generally used on GPU.
- **Single-Program-Multiple-Data (SPMD)**
	- The kernel contains loops and conditional instructions.
	- Each processing element has its own program counter.
	- Each instance of the kernel has a different execution flow.
	- Generally used on CPU.
- **Data-Parallel Programming Model**
	- The same sequential instruction stream is executed in lock-step on different data.
	- Generally used on GPU.
- **Task-Parallel Programming Model**
	- The program issues many kernels in an out-of-order fashion.


The platform model involves multiple compute devices, each comprising compute units and processing elements for parallel processing. OpenCL's program structure mirrors CUDA, with host code written in C++ for sequential tasks and device code in OpenCL C for managing kernels and data parallel execution.

![](images/Pasted%20image%2020240521213927.png)

OpenCL's memory model is comparable to CUDA's, including private, local, global, and constant memory types.
Different keywords, but same stuff of CUDA:

- Define an N-dimensional integer index space (N=l ,2,3) called NDrange
- Execute a kernel at each point in the integer index space

![](images/Pasted%20image%2020240521214245.png)

| OpenCL Term | Description                                                                      | CUDA Equivalent |
| ----------- | -------------------------------------------------------------------------------- | --------------- |
| Work-item   | Single execution of the kernel on a data instance (i.e., the basic unit of work) | CUDA thread     |
| Work-group  | Group of contiguous work-items                                                   | CUDA block      |
| NDrange     | N-dimensional grid with a local and global indexing, spanned by work-groups      | CUDA grid       |

The program object encapsulates both source and compiled code, which is compiled at runtime based on the selected device, using just-in-time compilation. Memory objects like buffers and images are employed for data transmission.

![](images/Pasted%20image%2020240521214629.png)

The context object encapsulates various features like command queues, unlike CUDA, where much is managed by the runtime due to its focus on a single accelerator

Just-in-time compilation in OpenCL provides flexibility across different devices, compiling device-side code at runtime using LLVM. 

### Workflow of an OpenCL Application

1. **Discovering the Platforms and Devices**
	- Platforms correspond to vendor-specific libraries, with functions like `clGetPlatformID` and `clGetPlatformInfo` helping identify available platforms. These functions retrieve information about platforms, such as names, vendors, and profiles. Each platform can have multiple devices.
	- Device IDs are selected for acceleration tasks using functions like `clGetDeviceID` and `clGetDeviceInfo`. 
1. **Creating a Context**
2. **Creating a Command Queue per Device**
3. **Creating Memory Objects to Hold Data**
4. **Copying the Input Data onto the Device**
5. **Creating and Compiling a Program from the OpenCL C Source Code**
6. **Generating a Kernel of the Program and Specifying Parameters**
7. **Executing the Kernel**
8. **Copying Output Data Back to the Host**
9. **Releasing the OpenCL Resources**
