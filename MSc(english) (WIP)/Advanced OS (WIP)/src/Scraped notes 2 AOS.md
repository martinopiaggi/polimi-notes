# PoliteCniCo Di Milano <br> Advanced Operating Systems 

Professor: VitTorio ZaCCARIA

Author: Riccardo Andrea IzZO

AY 2021/2022

JANUARY 2022

## Contents

1 Purpose, roles and architecture of an OS 1

1.1 Structure and roles of an OS . . . . . . . . . . . 1

1.2 Architecture . . . . . . . . . . . . . 6

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-03.jpg?height=51&width=1366&top_left_y=777&top_left_x=371)

2.1 Structure of linux user space programs . . . . . . . . . 7

2.2 Scheduling non-realtime processes . . . . . . . . . . 13

3 Multi-process programming and IPC 18

4 Task scheduling 19

4.1 Scheduling algorithms . . . . . . . . . . . . 19

4.2 Priority-based and multi-processor scheduling . . . . . . . 24

5 Virtual memory $\quad 26$

5.1 Virtual address space . . . . . . . . . . . . . 29

5.2 Physical address space . . . . . . . . . . . . 32

6 Threads and synchronization $\quad 35$

6.1 User space concurrency . . . . . . . . . . . . 35

6.2 Kernel space concurrency . . . . . . . . . . . 40

7 Drivers and IO $\mathbf{4 5}$

7.1 Device files . . . . . . . . . . . . . . 45

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-03.jpg?height=54&width=1304&top_left_y=1732&top_left_x=432)

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-03.jpg?height=57&width=1304&top_left_y=1779&top_left_x=432)

7.4 Peripheral Component Interconnect . . . . . . . . . 55

7.5 Storage devices ................ 56

8 File systems $\quad 58$

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-03.jpg?height=57&width=1301&top_left_y=2023&top_left_x=434)

8.2 Improving performance . . . . . . . . . . . 60

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-03.jpg?height=52&width=1304&top_left_y=2126&top_left_x=432)

9 Boot, ACPI and power management $\quad 61$

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-04.jpg?height=44&width=1293&top_left_y=493&top_left_x=438)

9.2 UEFI . . . . . . . . . . . . . . . . . . 63

9.3 ACPI and Device Tree ................ 64

$\begin{array}{ll}10 \text { Virtualization and hypervisors } & 65\end{array}$

## Purpose, roles and architecture of an OS

### Structure and roles of an OS

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-05.jpg?height=757&width=1304&top_left_y=722&top_left_x=432)

Figure 1: OS structure

- REGULATION: grant access to resources, decide CPU/memory usage

1. Privileged/unprivileged modes in CPUs
2. Interrupts: external events that force the kernel to take back control of the CPU
3. Exceptions: on execution of specific instruction (divide by zero, bad pointers, system calls)

1st goal: maximize CPU utilization by preempting processes Increase CPU utilization

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-06.jpg?height=111&width=656&top_left_y=514&top_left_x=951)

## Reduce Latency

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-06.jpg?height=280&width=678&top_left_y=736&top_left_x=946)

Figure 2: Sharing CPU

2nd goal: process scheduling, decide which process to run next

- Fairness: don't starve processes, balance process time execution
- Throughput: good performance overall
- Efficiency: minimize overhead of scheduler
- Priority: reflect processes priority
- Deadlines: must do X action by certain time (ex. play audio)

| Name | Target (Goal) | Where |
| :--- | :--- | :--- |
| FIFO | turnaround | Linux SCHED_FIFO |
| Round robin | res. time | Linux SCHED_RR |
| CFS | CPU fair share | Linux SCHED_CFS |
| MLFQ | res. time | Solaris, Windows, MacOS, BSD |

Figure 3: Process scheduling TURNAROUND: time between the arrival and the end of a process RESPONSE TIME: time between the arrival of the process and the first response produced

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-07.jpg?height=580&width=851&top_left_y=675&top_left_x=626)

Figure 4: Memory management

- STACK: temporary data storage, used by the functions (local variables, passing parameters)
- LIBRARIES: extra functionalities
- HEAP: dynamic data allocation (malloc ecc.)
- DATA: initialized data (variables)
- BSS: uninitialiazed data
- TEXT: code region

PROCESS: instance of a program in execution, isolate memory address space (cannot access variables or other memory areas), can contains one or more threads

Process Control Block (PCB): data structure associated to each process that resides in kernel space, track state of the process (program counter, registers, memory)

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-08.jpg?height=941&width=1138&top_left_y=451&top_left_x=561)

Figure 5: Process States

- MEDIATION: the operating system put itself between the applications and the hardware, must track what applications can and cannot use (example: can a process access this file?)

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-09.jpg?height=799&width=596&top_left_y=468&top_left_x=775)

Figure 6: File system

SYSTEM CALLS: a way to ask for a privileged service, application can invoke kernel through system calls

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-09.jpg?height=171&width=1158&top_left_y=1739&top_left_x=475)

Figure 7: System calls

FILE SYSTEM: set of mechanisms and policies to regulate access to persistent storage; it provides a common abstraction, protection and regulates space usage VIRTUAL FILE SYSTEM (VFS): common interface that performs file path resolution and dispatches the operation to the specific file system implementation associated with the device

I/O BLOCK LAYER: optimizes device access, translates and finalizes operations as $\mathrm{I} / \mathrm{O}$ request to device blocks

### Architecture

- Monolithic: single large kernel library compiled all at once, no privileged transition, no interfaces, faster, no modules at runtime (reboot necessary)
- Micro-kernel: all "non-essential" components of the kernel are implemented as system or user-level programs, increase overhead, device driver can be swapped out, minimal process and memory management
- Hybrid: similar to micro-kernel, run some services in kernel space and run device drivers in user space
- Library OS: services are provided in the form of libraries and compiled with the application


## Programs and processes

### Structure of linux user space programs

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-11.jpg?height=577&width=938&top_left_y=687&top_left_x=585)

Figure 8: Sections

SECTION: allocated by the linker, differentiation between different purposes of a program

SEGMENT: group of contiguos sections characterized by common properties, each segment is represented in one program header

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-11.jpg?height=399&width=1681&top_left_y=1741&top_left_x=211)

Figure 9: Lifecycle of a program COMPILER: produce an assembly program (main.s)

ASSEMBLER: produce an object file (main.o), at this stage there are no segments but only sections (addresses not yet decided). It doesn't know where data/code should be placed in the address space, assumes each section starts at zero. Thanks to additional sections it supports the linker with the symbol table (SYMTAB) and with the relocation table (RELA), it is also called relocatable file.

- SYMTAB: holds the name and the offset of each created object
- RELA: instruct the linker where to fix things up (rewrite references), this will be done by the linker once the address is known

LIBC: libraries that contains functions, collection of object files

LINKER: decide the initial address of the regions and create the executable (ELF files)

- Phase 1: coalesces all segments with the same name, determines the size of each segment and the resulting address, stores all global definitions in a global symbol table that maps the definition to its final virtual address
- Phase 2: ensure each symbol has exactly one definition, for each relocation lookup the symbol virtual address in the symbol table and fix references to reflect address

LINKER SCRIPT: used by the linker to understand where to put the sections

DYNAMIC SHARED LIBRARIES: manages shared objects, this allow to have a single copy on disk and don't actually include the libc code in the executable

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-13.jpg?height=903&width=1241&top_left_y=470&top_left_x=412)

Figure 10: Dynamic shared libraries

1. Global Offset Table: in principle there is an entry that contains the address of the function, this is done by the resolver
2. The function call in the application points to the Procedure Linkage Table (created by the linker and filled up by the dynamic linker), first it jumps to the address in the GOT that doesn't exists yet, finally the "jump resolver" instruction actually call the resolver and invocates the dynamic linker that load the address of the selected function in the GOT
3. At the end we will have the address of the function here in the GOT, next time it can be called directly
4. This fills up on demand the addresses that are needed from other object files through the resolver (associated to the dynamic linker), use memmap 5. These libraries must work whatever the address are assigned to, must be compiled with PIC (position independent code) flag

DYNAMIC LINKER: fixes the resolution at run-time LOADER: follow the instructions in the executable file and load them in memory

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-14.jpg?height=762&width=721&top_left_y=817&top_left_x=716)

Figure 11: ELF

EXECUTABLE AND LINKABLE FORMAT (ELF): is the way in which a compiled program is represented on disk, it supports the loading process

- program headers: describe common properties of section groups (segments)
- section headers: properties of each section

THREAD: processes that share memory KERNEL THREAD: standard processes that performs some operations in the background in kernel-mode, their address space is the entire kernel's one, they are schedulable and preemptable, child of kthreadd

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-15.jpg?height=816&width=1044&top_left_y=497&top_left_x=557)

Figure 12: Task states

CLONING: done with fork(), creates a child process that is a copy of the current task, it differs from the parent only in terms of PID and some signals are not inherited COPY ON WRITE: parent and child shares a single copy, if data is written a duplicate is made and each process receives a unique copy

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-16.jpg?height=840&width=1917&top_left_y=477&top_left_x=104)

Figure 13: Context switch

SCRATCH: machine register that point to the actual task_struct, can be accessed only in privileged mode. If preempt_count $=0$ the kernel is allowed to switch task. Function schedule() look for TIF_NEED_RESCHED flag while ret_from_exc setup the scratch register

### Scheduling non-realtime processes

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-17.jpg?height=635&width=2005&top_left_y=596&top_left_x=60)

Figure 14: Linux scheduling classes

SCHEDULING CLASS: set of functions that include policy-specific code for picking up tasks to run

RQ: run-queue, list of tasks

GROUP SCHEDULING: group of tasks that have the same CPU share value

Completely separated sets of processes with a certain priority $\pi$ :

Real-time processes $\pi \in[0,99]$; they belong to scheduling classes: srHFn_FIFn, STHFn_RR

Don-realtime processes (ऽCHFn_NORMAI) $100 \leq \pi(\nu) \leq 139$ which depends on a nice value $\nu \in[-20,+19]$ :

$$
\pi(\nu)=\max (100, \min (100+\nu-\beta+5,139))
$$

Figure 15: Original Linux Scheduler Priority: reflects the importance of a thread, relative urgency of a thread to be scheduled compared to others

Nice value: penalty that a thread can impose on itself regarding the use of the processing resources, higher niceness $->$ lower timeslice beta: bonus due to sleeping longer than a threshold

Real-time processes $\pi \in[0,99]$; they belong to scheduling classes: SCHED_FIFO, SCHED_RR

Non-realtime processes (SCHED_NORMAL) $100 \leq \pi(\nu) \leq 139$ which depends on a nice value $\nu \in[-20,+19]$ :

$$
\pi(\nu)=120+\nu
$$

Figure 16: Modern day Linux Scheduler (CFS)

Previous Linux scheduler problems:

- Context switch overhead varies with the niceness mix
- Variable fairness associated with a unit increment of niceness
- Enforcing time slices is hard

The solution is the Completely Fair Scheduler introduced in 2007 for nonreal time processes

For each process $p$, its time-slice is computed as:

$$
\tau_{p}=f\left(\nu_{0}, \ldots, \nu_{p}, \ldots, \nu_{n-1}, \bar{\tau}, \mu\right) \sim \max \left(\frac{\lambda_{p} \bar{\tau}}{\sum \lambda_{i}}, \mu\right)
$$

$\bar{\tau}$ is called schedule latency. Configurable, default $6 \mathrm{~ms}$.

- $\mu$ is called minimum granularity. Configurable, default $0.75 \mathrm{~ms}$.
- $\lambda_{i}\left(\nu_{i}\right)$ is the weight associated with a process, a sort of priority. It has exponential formula, i.e., $\lambda_{i}=k \times b^{-\nu_{i}}$ (current values $k=1024, b=1.25$ ) CFS: it does not directly assign timeslices to processes, it assign a proportion of the processor time depending on the load of the system. This proportion is affected by each process's nice value 'v' which acts as a weight, changing the proportion of the processor time each process receives. When a process becomes runnable, if it has consumed a smaller proportion of the processor than the currently executing process, it runs immediately.

The idea is to share processor time through a weighted average which depends on the weight of the process and two parameters:

- Targeted latency: is the overall time in which you would like that each of the processes has been given some time to work, decreasing it increases responsiveness at the expense of context switching time
- Minimum granularity: a floor on the timeslice assigned to each process

Accounting update: On each timer interrupt at time $t$, CFS updates the variable vruntime $\left(\rho_{i}\right)$ and sum_exec_time $\left(\epsilon_{i}\right)$ of the current process with the time elapsed since the last measurement $\Delta t=\left(t-t_{i}\right)$ :

$$
\epsilon_{i}+=\Delta t, \rho_{i}+=\frac{\Delta t}{\lambda_{i}}, t_{i} \leftarrow t
$$

## Accounting effect:

b When $\epsilon_{j}=\tau_{i}$ or the process blocks or a new process with lower $\rho$ becomes ready, the next process is selected to run;

- Typically, this is the one with the smallest $\rho$ taken from a red-black tree (a container of tasks sorted by $\rho$ ).

Figure 17: Time Accounting

Group Scheduling: administrator can create groups of tasks and assign a CPU share value to each of them, Part of the control groups(cgroup) feature of Linux. A cgroup associates a set of tasks with a set of parameters for one or more subsystems (cpu and cpuset) For simplicity, assume that $\epsilon_{i}$ starts from 0 as soon as the process is selected the first time within the schedule latency.

$\rho$ is a measure of the dynamic priority of the process and depends on the time it has consumed but also its weight. Higher-weighted task (identified by a higher $\lambda_{i}$ dividend) are put nearer the leftmost side but only if they don't starve lower priority ones.

Unblocked processes are assigned the minimum value of all $\rho$ 's in the rb tree.

Why using $\rho$ ? all processes have consumed their share $\tau_{p}$ only if they have reached the same amount of $\rho$ :

$$
\Delta \rho_{i}=\frac{\tau_{\rho}}{\lambda_{\rho}}=\frac{\bar{\tau} \lambda_{\rho}}{\lambda_{\rho} \sum \lambda_{j}}=\frac{\bar{\tau}}{\sum \lambda_{j}}
$$

In practice, it is easy to see if a process got less allocated time, even with different weights, because it will have a lower $\rho$

When there are $\mathrm{n}>1$ CPUs, making all CPUs do equal work while respecting relative weights can get tricky.

-> Can't balance on the the same number of threads

-> Can't balance on the total process weights of each run queue q

- Simple load balancing can result in threads being migrated across the machine without considering cache locality or non uniform memory access (NUMA)
- At boot Linux builds its own view of the processor and memory hierarchy into scheduling domains $\mathrm{D}^{*}$, sets of CPUs which share properties and scheduling policies
- CFS load balancer uses a hierarchical bottom to top strategy traversing the scheduling domains

The load balancer is triggered periodically for each core or when a core becomes idle after schedule. It will start from the current domain trying to pull-in tasks from other crowded cores to even out the average load. It does this in two steps: find_busiest_group () to find the busiest group and then find_busiest_runqueue () to find the actual run-queue in the group from. Balance over average load of run queue $\Omega_{q}$, where the load of task $i$ on $q$ is:

$$
\omega_{i, q}=\lambda_{i, q} \times \gamma_{i, q}
$$

$\gamma_{i, q}$ is the CPU usage

## Multi-process programming and IPC

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-22.jpg?height=453&width=580&top_left_y=608&top_left_x=428)

Process 1

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-22.jpg?height=455&width=345&top_left_y=607&top_left_x=1020)

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-22.jpg?height=455&width=328&top_left_y=607&top_left_x=1365)

Process 2

IPC mechanisms allow to exchange information between different processes with their own address space, some POSIX IPC components are:

- Signals: unidirectional communication method, no data transfer, asynchronous, can be sent by processes or by the OS
- Pipes (Unnamed): based on the producer/consumer pattern, unidirectional, data are written/read in a FIFO
- FIFO (Named Pipes): like unnamed pipes but based on special files created in the filesystem and not on file descriptors, no actual I/O is performed (reading/writing to a disk file)
- Message Queues: based on a priority queue, suitable for multiple readers and multiple writers, the status of the message queue is observable, all in /dev directory
- Shared Memory: based on memory mapping concept, it allows two processes to share a memory segment

Linux permissions: $\mathrm{R}(\mathrm{read})=4, \mathrm{~W}($ write $)=2, \mathrm{X}($ executable $)=1$; owner, group, others

## Task scheduling

### Scheduling algorithms

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-23.jpg?height=531&width=1559&top_left_y=716&top_left_x=283)

Figure 18: Scheduling conceptual diagram

PREEMPTION: operation for temporarily suspending the execution of a task in order to execute another task, performed via context switch, can be task-triggered or OS-triggered (usual)

I/O BLOCKING: the task voluntarily suspend their execution because waiting for I/O data, the CPU is available to run other tasks

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-23.jpg?height=357&width=1047&top_left_y=1792&top_left_x=520)

Figure 19: Task model - $\mathbf{a}_{\mathbf{i}}$ : arrival (request) time, time instant at which task is ready for execution and put into the ready queue

- $\mathrm{s}_{\mathrm{i}}$ : start time, time instant at which execution actually starts
- $\mathbf{W}_{\mathrm{i}}$ : waiting time, time spent waiting in the ready queue, $\mathrm{W}_{\mathrm{i}}=\mathrm{s}_{\mathrm{i}}-\mathrm{a}_{\mathrm{i}}$
- $\mathbf{f}_{\mathbf{i}}$ : finishing (completion) time, time instant at which the execution terminates
- $\mathrm{C}_{\mathrm{i}}$ : computation (burst/execution) time, amount of time necessary for the processor to execute the task without interruptions
- $\mathbf{Z}_{\mathbf{i}}$ : turnaround time, difference between finishing and arrival time, in case of preemption/suspension contains also the interferences from the other tasks, CPU-bound $\left(\mathrm{Z}_{\mathrm{i}}=\mathrm{W}_{\mathrm{i}}+\mathrm{C}_{\mathrm{i}}\right)$, I/O-bound $\left(\mathrm{Z}_{\mathrm{i}} \gg \mathrm{W}_{\mathrm{i}}+\mathrm{C}_{\mathrm{i}}\right)$,

Scheduling metrics:

- Process utilization: percentage of time the CPU is busy
- Throughput: number of tasks completing their execution per time unit
- Waiting time (avg): average time the tasks spent in the ready queue
- Fairness: fair allocation of the processor
- Overhead: amount of time spent in taking scheduling decisions and context-switches

PREEMPTIVE: running tasks can be interrupted by the OS at any time, responsive

NON PREEMPRIVE: once started a task is executed until its completion, minimum overhead, not responsive

STATIC: scheduling decisions are based on fixed parameters, known before task activation

DYNAMIC: scheduling decisions are based on parameters that changes at runtime

OFFLINE: scheduler executed on a set ok known tasks, must be static, very limited

ONLINE: scheduler executed at runtime OPTIMAL: based on an algorithm that optimize a given cost function, complex, high overhead

HEURISTIC: based on heuristic functions, faster

Scheduling algorithms:

- First-In First-Out (FIFO): tasks scheduled in order of arrival, nonpreemptive, very simple, not responsive


## Task set

| Task | $\mathrm{a}_{\mathrm{i}}$ | $\mathrm{C}_{\mathrm{i}}$ | $\mathrm{s}_{\mathrm{i}}$ | $\mathrm{f}_{\mathrm{i}}$ | $\mathrm{Z}_{\mathrm{i}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $\tau_{1}$ | 0 | 7 | 0 | 7 | 7 |
| $\tau_{2}$ | 2 | 4 | 7 | 11 | 9 |
| $\tau_{3}$ | 4 | 4 | 11 | 15 | 11 |
| $\tau_{4}$ | 5 | 1 | 15 | 16 | 11 |

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-25.jpg?height=214&width=995&top_left_y=1232&top_left_x=608)

- Shortest Job First (SJF): tasks scheduled in ascending order of computation time, non-preemptive, optimal, risk of starvation

Task set

| Task | $\mathrm{a}_{\mathrm{i}}$ | $\mathrm{C}_{\mathrm{i}}$ | $\mathrm{s}_{\mathrm{i}}$ | $\mathrm{f}_{\mathrm{i}}$ | $\mathrm{Z}_{\mathrm{i}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $\mathrm{\tau}_{1}$ | 0 | 7 | 0 | 7 | 7 |
| $\tau_{2}$ | 2 | 4 | 8 | 12 | 10 |
| $\tau_{3}$ | 4 | 4 | 12 | 16 | 12 |
| $\tau_{4}$ | 5 | 1 | 7 | 8 | 3 |

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-25.jpg?height=211&width=995&top_left_y=2120&top_left_x=608)

- Shortest Remaining Time First (SRTF): preemptive variant of $\mathrm{SJF}$, it uses the remaining execution time to decide which task to dispatch, responsive, risk of starvation, fair for long tasks


## Task set

| Task | $\mathrm{a}_{\mathrm{i}}$ | $\mathrm{C}_{\mathrm{i}}$ | $\mathrm{s}_{\mathrm{i}}$ | $\mathrm{f}_{\mathrm{i}}$ | $\mathrm{Z}_{\mathrm{i}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $\tau_{1}$ | 0 | 7 | 0 | 16 | 16 |
| $\tau_{2}$ | 2 | 4 | 2 | 6 | 4 |
| $\tau_{3}$ | 4 | 4 | 7 | 11 | 7 |
| $\tau_{4}$ | 5 | 1 | 6 | 7 | 2 |

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-26.jpg?height=214&width=998&top_left_y=1140&top_left_x=607)

- Highest Responsive Ratio Next (HRRN): select the task with the highest response ratio $\mathrm{RR}_{\mathrm{i}}=\left(\mathrm{W}_{\mathrm{i}}+\mathrm{C}_{\mathrm{i}}\right) / \mathrm{C}_{\mathrm{i}}$, non-preemptive, prevent starvation


## Task set

| Task | $\mathrm{a}_{\mathrm{i}}$ | $\mathrm{C}_{\mathrm{i}}$ | $\mathrm{s}_{\mathrm{i}}$ | $\mathrm{f}_{\mathrm{i}}$ | $\mathrm{Z}_{\mathrm{i}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $\tau_{1}$ | 0 | 7 | 0 | 7 | 7 |
| $\tau_{2}$ | 2 | 4 | 8 | 12 | 10 |
| $\tau_{3}$ | 4 | 4 | 12 | 16 | 12 |
| $\tau_{4}$ | 5 | 1 | 6 | 7 | 3 |

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-26.jpg?height=214&width=995&top_left_y=2121&top_left_x=608)

- Round Robin (RR): tasks are scheduled for a given time quantum q (the choice of the quantum value is critical), when this expires the task is preempted and moved back to the ready queue, preemptive (insertion in queue), fair and responsive, no starvation, worst timearound

Task set

$q=2$

| Task | $\mathrm{a}_{\mathrm{i}}$ | $\mathrm{C}_{\mathrm{i}}$ | $\mathrm{S}_{\mathrm{i}}$ | $\mathrm{f}_{\mathrm{i}}$ | $\mathrm{Z}_{\mathrm{i}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $\mathrm{\tau}_{1}$ | 0 | 7 | 0 | 16 | 16 |
| $\tau_{2}$ | 2 | 4 | 2 | 10 | 8 |
| $\tau_{3}$ | 4 | 4 | 6 | 15 | 11 |
| $\tau_{4}$ | 5 | 1 | 10 | 11 | 6 |

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-27.jpg?height=212&width=982&top_left_y=1203&top_left_x=517)

Task 0 started at $\boldsymbol{t}=\mathbf{4}$ and terminated at $\boldsymbol{t}=\mathbf{9}$.

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-27.jpg?height=114&width=1255&top_left_y=1713&top_left_x=430)

### Priority-based and multi-processor scheduling

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-28.jpg?height=654&width=1025&top_left_y=600&top_left_x=542)

Figure 20: Multi-level queue scheduling

Multi-level Queue Scheduling: different scheduling algorithm for each queue, first task picked from the topmost non-empty queue, queue scheduling is preemptive, risk of starvation

Multi-level Feedback Queue Scheduling: once activated a task is moved in the highest priority queue, when it expires it is moved to the next queue, how to solve starvation?

- Time slicing: each queue gets a maximum percentage of the available $\mathrm{CPU}$ time, if it expires tasks in this queue are no longer scheduled and the next queue is activated (even if the queue before is non empty)
- Aging: the priority of the task is increased as long as it spends time in the ready queue (avoid starvation)

Multi-processor scheduling:

- Single queue: all the ready task in the same global queue, simple design, fair, good CPU utilization, not scalable. - Multiple queues: a ready queue for each processor, scalable, more overhead $->$ need load balancing.

Load balancing: processors may be idle with waiting tasks in other queues (CPU utilization), reduce waiting/response times by moving tasks (Performance), usually performed via task migration:

- Push model: a dedicated task periodically checks the queues' lenghts and move tasks
- Pull model: each processor notifies an empty queue condition and picks tasks from other queues.

Work stealing: "steal" a task from other queue in case of a empty queue, concurrency problem, can't determine the right queue from which pick the task

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-29.jpg?height=193&width=677&top_left_y=1145&top_left_x=732)

Figure 21: Work stealing

Hierarchical queues: global queue dispatching tasks in local ready queues, scalable, complex to implement, good CPU utilization

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-29.jpg?height=371&width=1200&top_left_y=1739&top_left_x=468)

Figure 22: Hierarchical queues

## $5 \quad$ Virtual memory

PROTECTION: isolation, protect processes from one another

RESOURCE AMPLIFICATION: afford to processes more memory that what is available physically, store only most used pages in memory and less used in the swap space

PAGING: mapping during the access on the physical memory (VPN $\mid$ PFN)

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-30.jpg?height=588&width=656&top_left_y=899&top_left_x=737)

Figure 23: Linear paging implementation

## Page Table Entry

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-30.jpg?height=487&width=963&top_left_y=1849&top_left_x=581)

Translation done on each load/store instruction fetch, a page-table base register is used to get the PTE (page table entry)

TRANSLATION LOOKASIDE BUFFER (TLB): CPU-local associative cache that avoid accessing external memory, this reduce the time taken to access a user memory location. In particular it stores the recent translations of virtual memory to physical memory and can be called an address-translation cache.

How to avoid translations from some previously run process:

- Flush the TLB on context switches
- Use ASID (address space identifier) for each process field in the TLB, this uniquely identifies a process.

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-31.jpg?height=746&width=1353&top_left_y=1264&top_left_x=386)

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-32.jpg?height=903&width=1244&top_left_y=432&top_left_x=424)

Most OS have a high watermark (HW) and a low watermark (LW) to help decide when to start evicting pages from memory, when there are fewer than LW pages a background thread (swap/page daemon) manages to free memory.

Replacement policies:

- Belady's optimal: replaces the page that will be accessed furthest in the future, simple but difficult to implement
- FIFO
- LRU (least recently used): based on the principle of locality, if a program has accessed a page in the near past it is likely to access it in the near future. List based or timestamp based but the best way to implement it is the Corbatò's clock:
- introduce a reference bit for each page of the cache
- when a page is referenced (read/written) the bit is set to 1 - circular list of pages with an iterator, on a page request the OS checks the currently pointed page $\mathrm{P}$ 's refererence bit. If $\mathrm{P}=1$ it is sets to 0 and moves to $\mathrm{P}+1$ to recheck, if $\mathrm{P}=0$ the page is used as a new one.

DEMAND PAGING: the virtual address space starts out empty, bring in data in physical memory only when it is accessed, for this must track what logical pages have been allocated for each process.

All virtual pages are marked in the page table as not present. When accessing a virtual page that is not present, the CPU generates a page fault. The kernel determines what the content of the accessed page should be and then update the page table to mark the page as present.

WORKING SET: maximum number of pages for each process that the system can admit, avoid thrashing

### Virtual address space

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-33.jpg?height=683&width=683&top_left_y=1388&top_left_x=707)

Figure 24: Linux virtual address space Each process has its own page directory. Kernel logical (kmalloc) addresses are mapped directly to physical memory starting from 0 while kernel virtual (vmalloc) ones are not contiguous

VIRTUAL MEMORY AREA (VMA): contiguous range of virtual addresses, PTE doesn't represent all the pages of a process but only the ones accessed, understand if a VPN is valid but not mapped. Handler looks for a VMA in vma_list, VMAs can be created explicitly by a process with mmap().

mm_struct: pointed by the mm memory descriptor (field in task_struct). It stores the start and the end of memory segments, the number of physical memory pages used by the process (rss: Resident Set Size) and the amount of virtual space address space used.

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-34.jpg?height=1017&width=1131&top_left_y=1096&top_left_x=497)

Figure 25: Virtual Memory Area VMA can be classified as:

- Mapped to file (with backing store: memory area that comes from PT_LOAD in ELF file) or not mapped (anonymous i.e. stack, heap)
- Shared or private
- Readable, writable or executable

1. Program calls brk() to grow its heap

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-35.jpg?height=299&width=566&top_left_y=1035&top_left_x=297)

3. Program tries to access new memory. Processor page faults.

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-35.jpg?height=296&width=751&top_left_y=1511&top_left_x=297)

2. brk() enlarges heap VMA.

New pages are not mapped onto physical memory.

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-35.jpg?height=295&width=686&top_left_y=1034&top_left_x=923)

4. Kernel assigns page frame to process, creates PTE, resumes execution. Program is unaware anything happened.

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-35.jpg?height=296&width=569&top_left_y=1508&top_left_x=1098)

Figure 26: VMA - Demand paging

When the program asks for more memory via the brk() system call, the kernel simply updates the heap VMA. No page frames are actually allocated at this point and the new pages are not present in physical memory. Once the program tries to access the pages, the processor page faults and do_page_fault() is called. Now the kernel assign the page frame to the process and creates the PTE entry.

### Physical address space

All the physical pages in the system are represented by a struct page in an array mem_map, memory may be arranged in NUMA banks or nodes.

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-36.jpg?height=417&width=759&top_left_y=735&top_left_x=672)

Linux uses a node-local allocation policy to allocate memory from the node closest to the running CPU. A zone is a memory range within a 'node':

- ZONE_DMA: memory in the lower physical memory ranges which certain ISA devices require
- ZONE_NORMAL: directly mapped by the kernel into the upper region of the linear address space
- ZONE_HIGHMEM: the remaining available memory in the system and is not directly mapped by the kernel

Each zone contains the total size of pages in the zone and an array of lists of free page ranges. When kernel needs contiguous pages and there are free pages, the buddy algorithm is used to choose how to split free memory regions to satisfy the request.

BUDDY ALGORITHM: tries avoiding as much as possible the need to split up a large free block to satisfy a request for a smaller one, easily allow to merge back blocks when they are freed.

PAGE FRAME: unit of physical memory management PAGE CACHE: set of physical pages that are the result of read/write of regular filesystem files, stored in address_space, maps file descriptor + offset $->$ physical page but works also in physical page $->$ VMA

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-37.jpg?height=421&width=706&top_left_y=430&top_left_x=616)

Figure 27: Page descriptor

_mapcount $=$ how many sharings there are (file-backed page of files shared between processes), address_space $=$ how to reach them in case we want to invalidate them (CoW pages)

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-37.jpg?height=740&width=1179&top_left_y=1338&top_left_x=451)

Figure 28: Page frame reclaim algorithm PAGE FRAME RECLAIM ALGORITHM: tries to relieve memory pressure caused by file-backed pages in physical memory that changes dinamically. Keep two lists of clean pages for each zone, active and inactive. Victims are taken from the inactive list and must be not dirty, pages go into the active list only after two accesses. Linux periodically moves the pages from active to inactive trying to keep lists balanced.

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-38.jpg?height=471&width=1073&top_left_y=854&top_left_x=488)

Figure 29: Slab allocator

SLAB: intermediary structure which consists of one or more contiguous page frames that contain both allocated and free objects When the kernel needs to allocate a new task_struct, it looks in the object cache for task structures, and first tries to find a partially full slab SLAB ALLOCATOR: cache commonly used objects so that the system doesn't waste time allocating/deallocating, allocation of small objects of memory to eliminate internal fragmentation

## Threads and synchronization

### User space concurrency

Race condition: the results depend on the timing execution of the code itself Critical section: a piece of code that accesses a shared variable and must not be concurrently executed by more than one thread

Mutual exclusion: if one thread is executing within the critical section, the others will be prevented from doing so

Locks: ensure that any critical section executes as if it were a single atomic instruction, type of locks:

- interrupted-based: disable and enable interrupts, work well on uniprocessor systems

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-39.jpg?height=315&width=534&top_left_y=1167&top_left_x=791)

- spin-locks: used at kernel level, based on atomic exchange, with or without hardware support, spin-waiting wastes time by waiting actively for another thread to release a lock

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-39.jpg?height=525&width=1035&top_left_y=1820&top_left_x=537)

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-40.jpg?height=772&width=1730&top_left_y=564&top_left_x=187)

- yielding: it deschedules itself and puts itself at the end of the ready queue

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-40.jpg?height=314&width=1029&top_left_y=1699&top_left_x=536)

- queue-based sleeping: based on a wait queue in the kernel, locking a contended lock puts the process in wait, unlock will wakeup a single process from the wait list

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-41.jpg?height=360&width=1403&top_left_y=431&top_left_x=341)

Condition variable: explicit queue where thread can put themselves on, pthread_cond_wait() and pthread_cond_signal() that wakes up one thread Semaphore: used as both locks and condition variables. sem_wait(s) decrements the value of semaphore $\mathrm{s}$ by one and wait if the value is negative (= number of waiting threads in this case), sem_post(s) increments the value of semaphore $\mathrm{s}$ and wakes one of the threads waiting

Reader/writer locks: the first and the last reader get/release an additional lock, the write lock, this allow many lookups to proceed concurrently while serializing inserts

Concurrency properties:

- Safety (correctness): we don't reach an error state or work on invalid data
- Liveness (progress): we eventually reach a final state

Priority inversion: lower priority tasks are preventing an higher priority one to execute

Techniques used for solving priority inversion, setup with pthread_mutexattr_setprotocol() (based on changing the priority of the tasks):

- Highest locker priority (HLP): avoid preemption during the execution of any critical section by raising the priority of task accessing the shared resource

$$
p_{i}\left(R_{k}\right)=\min _{h}\left\{P_{h} \mid T_{h} \text { using } R_{k}\right\}
$$

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-42.jpg?height=569&width=1477&top_left_y=618&top_left_x=324)

- Priority inheritance (PIP): when task-i enters the critical section and the shared resource is already held by task-j, task-j assumes the active priority of task-i. Problem: a higher priority task might end up ceding its priority to several tasks if it accesses multiple semaphores

$$
p_{j}\left(R_{k}\right)=\left\{P_{j}, \min _{i}\left\{P_{i} \mid T_{i} \text { blocked on } R_{k}\right\}\right\}
$$

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-43.jpg?height=607&width=1575&top_left_y=607&top_left_x=275)

- Priority Ceiling (PCP): task-i is allowed to enter a critical section only if its priority is higher that all the priority ceiling (highest priority among the tasks that can lock it) of the semaphores currently locked by other tasks

Deadlock: no task can take action because it is waiting for another task to take action, these conditions are:

- Mutual exclusion: two threads can't act on the same resource at the same time, can be prevented by using powerful hardware instructions without using locks
- Hold-and-wait: threads hold resources allocated while waiting for additional resources, can be prevented if a task voluntarily release the resource if the acquisition of further resources fails (use trylock())
- No preemption: resources cannot be forcibly removed from threads that are holding them, can be prevented like hold-and-wait
- Circular wait: circular chain of waiting threads, can be prevented by introducing ordering so that no cyclical wait arises (use total store ordering or partial ordering) Futex: fast locks, avoid system calls, unnecessary context switches and thundering herd problems (all threads waiting are woken up)

Uncontended case (futex): avoid system calls, requires a shared state in user space accessible to all partecipating processes/task and user-space atomic ops

Contended case (futex): application needs to wait in kernel for the release of the lock or needs to wake up a waiting task in the case of an unlock operation, a kernel object is required

Event-based concurrency: wait for an event to occur, manage the type of event received, rinse and repeat.

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-44.jpg?height=265&width=508&top_left_y=1017&top_left_x=797)

When an handler process an event it is the only activity taking place in the system, no locking needed and no context switch, use select or poll APIs. Problem: non-blocking I/O, event loop blocks and system sits idle, can be solved by running blocking file I/O operations in a thread pool.

### Kernel space concurrency

Kernel preemption: in a preemptive kernel at the end of every interrupt the scheduler is called for context switch, multiple threads running in kernel mode access shared structures Jiffies $=$ machine uptime, number of clock ticks that have occured since the system booted. It is incremented by one during each interrupt, read by process context code within the kernel itself.

Interrupt: can occur asynchronously at almost any time, interrupting the currently executing code in kernel mode.

Multiprocessing: kernel code must simultaneously run on two or more processors, they can simultaneously access shared data at exactly the same time - SMP-safe: code that is safe from concurrency on symmetrical multiprocessing machines, code that will execute correctly even in the face of concurrency access from multiple thread of execution

- Sequential consistency: any execution is the same as if the operations of all the processors were executed in some sequential order, the operations of each processor appear in this sequence in the order specified by its program
- Partial store order: each processor read/write to its own complete copy of the memory while each writes propagates independently with reordering allowed as the writes propagates, writes can be reordered, weaker than TSO
- Total store order (TSO): use local write queue to hide memory latency, at the moment that a write reaches shared memory, any future read on any processor will see it and use that value, loads appears reordered with respect to stores, processors allow the use of specific "memory barriers" instructions, each thread flushes its previous write to memory before starting its read -> this force sequentially consistent behaviour at critical moments in a program

In linux kernel preemption is managed through the preempt_count in the thread_info struct of the current process, this is checked at preemption points; a non-zero counter tells to the kernel that it cannot perform a context switch. The variable is incremented every time a thread acquires a lock in kernel mode and whenever the kernel is beginning the execution of an interrupt.

Type of locks in kernel:

- Spinning locks: single-holder lock, if you can't get the spinlock you keep trying until you can. acquire(), acquire_inc()(memory fence instruction)
- Sleeping locks: semaphores in Linux, when a task attempts to acquire a semaphore that is unavailable, the semaphore places the task onto a wait queue and puts the task to sleep. Spinlocks before accessing the shared lock. acquire_sleep(), release_sleep () , sleep () (atomically release lock and sleep on channel, requires lock when awakened), wakeup()(wake up all processes sleeping on channel) - Spinlocks: on uniprocessor machines these locks become simple calls to preempt_disable(), use spin_lock_irqsave() and spin_unlock_irqrestore to disable local interrupts

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-46.jpg?height=214&width=559&top_left_y=635&top_left_x=765)

- Readwrite locks: similar to spinlocks

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-46.jpg?height=421&width=949&top_left_y=1091&top_left_x=570)

- Seqlocks: useful if we have a structure with many readers and few writers, avoid writers from being starved. The idea here is to speculate that a race doesn't occur, detect it, and retry if it does

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-46.jpg?height=472&width=1677&top_left_y=1861&top_left_x=210)

- Ordering and barriers: macros to reorder reads and writes, wmb()
- Read copy update locks (RCU): low latency read on shared data that is more frequently read than written, readers avoid locks and tolerate concurrent writes, writers create a copy of data structure and publish new one with a single atomic instruction

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-47.jpg?height=786&width=989&top_left_y=954&top_left_x=557)

Destructive operation in two parts: in the first part the list is modified in an atomic way, in the second part the data block is reclaimed and freed.

- Queued Spinlocks (MCS locks): in SMP systems every attempt to acquire a lock requires moving the cache line containing that lock to the local CPU, for contended locks this cache-line bouncing can hurt performance significantly.

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-48.jpg?height=450&width=1011&top_left_y=463&top_left_x=557)

## Drivers and IO

### Device files

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-49.jpg?height=653&width=800&top_left_y=709&top_left_x=649)

Figure 30: Modern system architecture

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-50.jpg?height=797&width=762&top_left_y=437&top_left_x=688)

Figure 31: Canonical device interface

Command Block Registers aka I/O Port Space contains the addresses where it is possible to read/write

Direct Memory Access (DMA): standard in modern architectures, it allow peripherals to access internal memory and exchange data (read/write) without calling the CPU by generating an interrupt for every transfered block Device are addressable through:

- I/O Ports: provides explicit I/O instructions, in/out privileged instructions can be used to communicate with devices, each device is assigned a port number in the $\mathrm{I} / \mathrm{O}$ address space which names the device
- Memory-mapped I/O: the hardware makes device registers available as if they were memory location

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-51.jpg?height=309&width=1241&top_left_y=436&top_left_x=431)

Figure 32: Device drivers

Storage device: addressable stores of data blocks (sectors), special files into the file system in / dev directory

- character device: character stream, read/write has direct impact on the device itself
- block device: sequence of numbered blocks, each block can be individually addressed and accessed

Each device has a special driver that handles it, each driver has a major device number that identify it, if a driver supports multiple devices a minor device number is used to identify them

Driver: set of programs that converts higher-level requests for data into device-specific requests for sectors' data

buf: intermediate structure, generic container for sectors' data

Interrupt Service Routine (ISR): manage interrupts

LBA: logic block address scheme, location of blocks of data stored on storage devices

IDE interface: queues a request if there are others pending or issues it directly to the disk, when the request is complete the calling process is put to sleep; wait the driver to be ready by reading the Read Status Register (0x1F7) until it is READY and not BUSY.

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-52.jpg?height=499&width=1637&top_left_y=474&top_left_x=255)

Figure 33: Char devices

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-52.jpg?height=456&width=1139&top_left_y=1298&top_left_x=512)

Figure 34: Char devices anatomy

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-53.jpg?height=759&width=1589&top_left_y=463&top_left_x=260)

Figure 35: Block devices

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-53.jpg?height=669&width=1266&top_left_y=1498&top_left_x=408)

Figure 36: Block devices anatomy

## $7.2 \quad$ Interrupts

Interrupts are issued by interval timers and I/O devices; for instance, the arrival of a keystroke from a user sets off an interrupt.

Exceptions, on the other hand, are caused either by programming errors or by anomalous conditions that must be handled by the kernel.

Asynchronous I/O interrupts: generated by other hardware devices at arbitrary times with respect to the CPU clock signal

- Maskable interrupts: all interrupt requests (IRQs) issued by I/O, is ignored by the control unit
- Nonmaskable interrupts: always recognized by the CPU, non-recoverable hardware errors

Synchronous interrupts (exception): produced by the CPU control unit, the control unit issues them only after terminating the execution of an instruction

- Fault exception: correct and re-execute the faulting instruction
- Trap: don't re-execute excepting instruction

Interrupts or exceptions are identified by a number ranging from 0 to 255 (vector).

The interrupt descriptor table contains handlers for all vectors, it is pointed by the idtr machine register, entries identifies a segment in memory.

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-55.jpg?height=889&width=1220&top_left_y=455&top_left_x=431)

Figure 37: I/O interrupt handling flow

IRQ number is $n$ while vector is $32+n$, interrupt[n] is pointed to by the IDT entry. It saves the IRQ value and the register's contents on the Kernel Mode stack.

do_IRQ(n) sends an ack to the PIC that is servicing the IRQ line. Every IRQ has its own irq_desc_t descriptor. handler points to the PIC object (hw_irq_controller descriptor) that service the IRQ line, action identifies the interrupt service routine to be invoked when the IRQ occurs.

Programmable Interrupt Controller (PIC): monitors the IRQ lines checking for raised signals, if two or more IRQ lines are raised selects the one having the lower pin number.

Interprocessor Interrupt (IPI): delivered directly as a message on the bus that connect the local ACPI of all CPUs (not through IRQ line)

- CALL_FUNCTION_VECTOR: sent to all the CPUs but the sender, force all these CPU to run a function (example: stop) - RESCHEDULE_VECTOR: force a CPU to reschedule its task
- INVALIDATE_TLB_VECTOR: sent to all the CPUs but the sender, force them to invalidate their Translation Lookaside Buffers


### Deferring work

Idea: move the non-critical management of interrupts to a later time

- Top half: works in an non-interruptible scheme and schedules some deferred work (deferred functions)
- Bottom half: finalizes the work by deferred functions from queue and executing them, invoked in reconciliation points where it looks for tasklet that are scheduled but not running

SoftIRQs: deferred functions, statically allocated at compile time, managed in two phases: activation (raised by an interrupt handler) and execution (at specific reconciliation points). Linux ensures that when a softirq is run on a $\mathrm{CPU}$, it cannot be preempted on that CPU but it can run on another CPU with the same handler.

Tasklet: softIRQs type, no more than one instance of the same tasklet can run over all the CPUs, function plus some data, they cannot sleep and must use spinlocks that disable the interrupts

Work queue: schedulable entity that runs in process context to execute the bottom half, general mechanism to submit work to a worker kernel thread. The worker thread enters an infinite loop and goes to sleep. When work is queued, the thread is awakened and processes the work. When there is no work left to process, it goes back to sleep.

Types of timers:

- System timer: a programmable piece of hardware that issues an interrupt at a fixed frequency (tick rate), the time interrupt performs periodic work such as update the timeslice of a process
- Dynamic timer: schedule events that run once after a specified time has elapsed - Real-time clock (RTC): nonvolatile device for storing the system time, it continues to keep track of time even when the system is off by way of a small battery typically included on the system board

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-57.jpg?height=1008&width=542&top_left_y=672&top_left_x=802)

Figure 38: Linux block layer

- VFS: locates the disk and the file system hosting the data starting from a path, verifies if the data is already mapped in memory (kernel's page cache)
- Mapping layer: accesses the file descriptor and pieces together the logical-to-physical mapping, get the position of the actual block
- Block I/O layer: creates a list of I/O structures (called bio: describes one or more contiguous segments of data), each representing an $\mathrm{I} / \mathrm{O}$ operation (actual request) to be submitted to the disk and sends them to the device driver
![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-58.jpg?height=828&width=1590&top_left_y=648&top_left_x=344)

Figure 39: Linux block IO operation

The Block Layer receives bio requests, each indicates the sector and what data is requested within that sector. Block devices maintain a hardware request queues to store their pending block I/O requests, it is represented by the request_queue structure. Driver's queue_rq is invoked through a mechanism called plugging that adjusts the rate at which requests are dispatched to the device driver.

IO scheduling policies:

- NOOP: efficiency, global I/O requests queue ordered using FIFO, the scheduler merges requests to adjacent sectors to maximize throughput
- Complete Fair Queuing: assign to each process a fair slice of the disk bandwidth, I/O requests queue for each process, the scheduler picks bunch of $\mathrm{I} / \mathrm{O}$ requests using $\mathrm{RR}$ - Deadline: prioritize long starving reads, the scheduler assign an expiration time to any incoming I/O requests, two FIFO queues for requests + two sector-wise sorted queues for merging requests
- Anticipatory: improve deadline I/O, applies an anticipatory heuristic, the scheduler waits for a few ms speculating on the chance of receiving subsequent adjacent requests
![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-59.jpg?height=998&width=1374&top_left_y=859&top_left_x=360)

Figure 40: I/O scheduling policies

### Peripheral Component Interconnect

Peripheral Component Interconnect (PCI): complete set of specifications defining how different parts of a computer should interact, supports autodetection of interface boards (automatically configured at boot time). Each PCI peripheral is identified by a bus number, a device number, and a function number. At system boot, the firmware performs configuration transactions with every PCI peripheral inorder to allocate a safe place for memory mapped IO / Port IO regions it offers, the result is stored in an array of data structures called Base Address Registers (BARS) that contains a unique function ID and the interrupt line. The kernel will run your driver probe callback either at boot time or in response to a hot-plug event when compatible hardware is found. It automatically passes a pci_devstruct.

### Storage devices

Given a set of block requests is useful decide which one to schedule next to minimize turnaround time

SSD: no moving part, based on flash technology, to write to a given chunk of it, you first have to erase a bigger chunk, writing too often cause wear out. The standard storage interface is a simple block-based one, where blocks (sectors) of size 512 bytes (or larger) can be read or written given a block address. An SSD also contains some amount of volatile (i.e., non- persistent) memory (e.g., SRAM); such memory is useful for caching and buffering of data.

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-60.jpg?height=325&width=1499&top_left_y=1537&top_left_x=302)

Flash translation layer: takes read/write requests on logical blocks and turns them into low-level read, erase and program commands, minimize write amplification (erase entire block to write only a page), minimize program disturbance (some bits get flipped when accessing), apply wear leveling (distribute program operations to all the pages, spread work across all the blocks) - Simple FTL: map logical page N to physical page N, increase write amplification and wear out

- Log structured FTL: the device appends the write to the next free spot in the currently-being-written block, keeps a mapping table that stores the physical address of each logical page in the system
- Hybrid mapping: reduce the size of the mapping table, the FTL keeps a few blocks erased and directs all writes to them; these are called log blocks.

Garbage collection: find a block that contains one or more garbage pages, read in the live (non-garbage) pages from that block, write out those live pages to the log and finally reclaim the entire block for use in writing.

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-61.jpg?height=777&width=745&top_left_y=1254&top_left_x=251)

4 erases

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-61.jpg?height=864&width=702&top_left_y=1319&top_left_x=1161)

## $8 \quad$ File systems

### EXT file system organization

Given the inode table base block $T$ (from the superblock), we can compute the block containing inode $i$ as:

$$
b(i)=T+\left(i>>\left(s_{b}-s_{n}\right)\right)
$$

where $2^{s_{n}}$ is the size of the inode in bytes, while $2^{s_{b}}$ is the size of a block in bytes.

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-62.jpg?height=263&width=1021&top_left_y=1083&top_left_x=514)

Figure 41: inode table

- inode table: contains a listing of all inode numbers for the respective file system. When users search for or access a file, the UNIX system searches through the inode table for the correct inode number
- inode: 256-byte block on the disk that stores data about the file. This includes the file's size, user IDs of the file's user and the group owner, the access permissions. It finally contains data that points to the location of the file's data on the hard drive
- superblock: contains the metadata that defines the other filesystem structures and locates them on the physical disk, it also has the number of inodes and where the inode table begins
- i-bmap: how many have been used or not, $0=$ free and $1=$ used
- d-bmap: usually free or used for something else A directory has an inode, somewhere in the inode table (with the type field of the inode marked as "directory" instead of "regular file").

JOURNALING: mechanism that keeps track of changes not yet committed to the file system's main part, in the event of a system crash/power failure the file system can be brought back online more quickly with a lower risk of being corrupted. Before overwriting the structures in place first write down a little note describing what you are about to do (write ahead), if a crash takes place you can go back to the notes and you will know exactly what to fix instead of having to scan the entire disk.

- journal write: write the transaction, including a transaction-begin block, all pending data and metadata updates, and a transaction-end block, to the log; wait for these writes to complete
- checkpoint: write the pending metadata and data updates to their final locations in the file system

Once this transaction is safely on disk, we are ready to overwrite the old structures in the file system; this process is called checkpointing.

Redo logging: the file system recovery process will scan the log and look for transactions that have committed to the disk, these will be replayed in order.

Journaling file systems treat the $\log$ as a circular data structure (circular $\log$ ) that can be used many times -> problem: we write each data block to the disk twice, solution: metadata journaling, writes before anything else the data in the final location, this results in improved performance at the expense of increased possibility for data corruption.

Types of journaling:

- journal: full data journaling, writes twice on the disk, reliable but slow
- ordered: writes data blocks first, then journal the metadata, terminates the transaction and writes the metadata, consistent state
- writeback: metadata journaling with no strict sequencing of writing the data blocks first, fastest
- ZFS: never overwrites files or directories in place; rather it places new updates to previously unused locations on disk. After a number of updates are completed, COW file systems flip the root structure of the file system to include pointers to the newly updated structures
- Backpointer-based consistency (BBC): each data block has a reference to the inode to which it belongs. The file system can determine if the file is consistent by checking if the forward pointer (the address in the inode or direct block) points to a block that refers back to it)


### Improving performance

FFS: divide the disk into a number of cylinder groups, this aggregates $\mathrm{N}$ consecutives cylinders into a group $->$ keep related stuff together

Improve speed: cache aggressively, perform all operations in memory and leave them in memory until the buffer cache gets flushed Improve space management: extent-based approach, it maps a logical file block range to a physical block range, compact and increase performance when read/write large file (pointer-based less efficient in this case)

Log structured file system (LFS): buffer all updates in an in-memory segment, when the segment is full it is written to disk in one long sequential transfer to an unused part of the disk, sort of copy on write

### Linux VFS

Implements the file and filesystem-related interfaces provided to user-space programs, enables programs to use standard Unix system calls to read and write to different filesystems. The VFS is object-oriented with the structures that contain both data and pointers to filesystem-implemented functions that operate on the data.

- Static informations: corresponding to the one stored in the block device
- Dynamic informations: associated with files and directories opened so far


## Boot, ACPI and power management

### BIOS

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-65.jpg?height=675&width=705&top_left_y=682&top_left_x=707)

Figure 42: BIOS

BIOS: CPU is reset at startup, processor starts in real mode (20-bit of address space accessible, caches and MMU disabled).

Master Boot Record (MBR): contains the sequence of instructions necessary for the boot; holds the information on how the logical partitions (partition tables), containing file systems, are organized on that medium.

BIOS operations:

1. Copies itself into RAM
2. Looks for video adapters
3. Start Power On Self Test: testing devices, initializes video card, does a RAM consistency check
4. Boot configuration loaded from the NVRAM 5. Identify the stage 1 bootloader loading a sector (MBR) form disk and jumps to it

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-66.jpg?height=971&width=1019&top_left_y=626&top_left_x=572)

Figure 43: Bootloader stages (MBR)

- First Stage Boot Loader (FSBL): sets up the stack, switch to 32-bit protected mode and read the kernel code (software image from flash memory)
- 1.5SBL: code that read/write the file system
- SSBL: reads a configuration file (boot selection menu), the kernel initial image is loaded in memory
- Advanced Configuration and Power Interface (ACPI): describe the hardware configuration, tells where the PCI tables are placed in the memory - normal.mod: module that provides the classical interface of GRUB
- grub.conf: how many operating systems?


### UEFI

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-67.jpg?height=970&width=1019&top_left_y=797&top_left_x=561)

Figure 44: UEFI

Replacement of BIOS, overcome its size limitations, modular (can be extended with drivers), takes control right after the system is powered on and loads the firmware settings into RAM, startup file stored in a FAT32 partition, no need for MBR code (only used for compatibility).

GUID Partition Table (GPT): standard for the layout of partition tables on a disk, modern way in which disks are partitioned, stored in the first 33 blocks of the disk

### ACPI and Device Tree

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-68.jpg?height=712&width=805&top_left_y=739&top_left_x=649)

Figure 45: ACPI

ACPI: open standards for OS, configure computer hardware components, perform power management, auto configuration and status monitoring. Kernel receives a platform description in terms of tables that contain code and reference registers, code is executed with an interpreter. ACPI defines a way to describe the power consumption of a system, particular configuration of states. Tables are discovered by the kernel by looking for a string in memory in two memory areas where the BIOS has copied them, they specify a graph of properties and methods called the namespace.

The namespace is a hierarchical data structure that describes the underlying hardware platform. The main table is the Differentiated System Description Table (DSDT) and describes devices and their IRQ mappings.

- AML Interpreter: interpreter of AML, a binary language that contains both data and code (stored in the UEFI firmware) - OSPM: operating system power manager
- acpid: ACPI daemon, list for ACPI events and triggers action of the OSPM, capture system control interrupts (SCI) to handle hardware events trigger

Device tree: data structure that describes hardware, have binary format for OS and textual format for management, useful when OS cannot detec$\mathrm{t} /$ probe this informations. Nodes are organized in a hierarchy as a collection of property and value tokens. Each addressable device gets an address space region (reg), interrupt signals are expressed as links between nodes independent of the tree. When a device is attached two values are associated to it: chip select value and offset (both added to the device tree). To get a usable memory mapped address the device tree must specify how to translate addresses from one domain (provided by ranges).

## Virtualization and hypervisors

Virtual machine: an efficient, isolated duplicate of a real computer machine

- Consolidate and partition hardware
- Horizontal scalability: react to variable workloads
- Standardized infrastructure: isolated network and storage
- Security

Virtualization requirement:

- Fidelity: equivalence of behaviour with the real machine
- Safety: the virtual machine cannot override the control of virtualized resources
- Efficiency: minor decrease in speed


## - $(\mathrm{LO}, \mathrm{B} 0)=$ memory area assigned to the $\mathrm{VM}$ <br> - $(\mathrm{L} 1, \mathrm{~B} 1)=\mathrm{L}$ and $\mathrm{B}$ registers of the $\mathrm{VM}$ <br> - $(\mathrm{L} 01, \mathrm{~B} 01)=\mathrm{L}$ and $\mathrm{B}$ register while the $\mathrm{VM}$ runs: (L0+L1, $\min (B 1, B 0-L 1)$ )

![](https://cdn.mathpix.com/cropped/2023_09_16_d0650f6cea4228b0f4a8g-70.jpg?height=253&width=1377&top_left_y=833&top_left_x=339)

$\mathrm{L}=$ lowest accessible address

$\mathrm{B}=$ virtual memory size

$\mathrm{S}=$ physical memory size

$\operatorname{read}(\mathrm{A}):$

if $\mathrm{A}+\mathrm{L}>\mathrm{B}->$ error

if $\mathrm{A}>\mathrm{S}->$ error

read from memory address $\mathrm{A}+\mathrm{L}$

- Control-sensitive instruction: affect the availability of resources, affect safety
- Behaviour-sensitive instructions: behave differently depending on the configuration of the machine, affect fidelity

Trap: L and B registers are extended to the full memory, mode is changed to supervisor, program counter is reset and execution starts from this point.

Privileged instructions always trap in user mode, an efficient VMM is possible if all sensitive instructions are privileged.

Trap-and-emulate virtualization: the VM always runs in user mode, it maintains a copy of privileged CPU registers (processor flags and segmentation registers), catch user->supervisor traps

Paravirtualization: replace the processor interface with a similar but faster software interface, require hypercalls (avoid fetch/decode costs and combine multiple traps into one). Fast read-only access to hypervisor (third mode in addition to user/supervisor) data (replace hardware registers with memory and allow some behaviour-sensitive instructions).

Obey Popek-Goldberg rule to improve performance: reduce number of traps, avoid shadow paging overhead, only trap on sensitive instructions, virtualize sensitive instructions and priviliged registers.

KVM: driver that add a character device that exposes the virtualization capabilities to userspace, a process can run a virtual machine in a fully virtualized PC. Each virtual machine is a process on the host, a virtual cpu is a thread in that process.

Advantages of KVM: reuse Linux development tools, don't have to write a full microkernel (reuse OS scheduler), flexible userspace implementation Disadvantages of KVM: larger API surface, microkernel vs monolithic OS, is Linux ready for VM workloads?
