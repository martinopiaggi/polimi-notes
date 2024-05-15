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