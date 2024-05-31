# Process

First, let's clarify some definitions:

- "**Task**" refers to a single unit of computation and is often used interchangeably with "thread" in Linux.
- A **program** is a set of computer instructions stored and not currently being executed. A **process** is an instance of a program that is currently being executed and has its own isolated memory address space. It has a unique program counter called `PID`, two stacks (one in user mode and one in kernel mode), a set of processor registers, and an optional address space.
- A **process** is an instance of a program that is currently being executed. It has its own isolated memory address space, a unique program counter called `PID`, two stacks (one in user mode and one in kernel mode), a set of processor registers, and an optional address space.
- An **application** is a user-oriented concept of a program, often referring to programs with a graphical user interface (GUI).
- A **thread** is the smallest schedulable unit of execution which can be **contained in a process**.
	- Threads have a unique PID and PPID (parent's PID)
	- Each thread in a process has its own stack for independent execution, while they share the process's global memory.
	- Variables that are part of the parent process's memory (e.g., global or static variables and heap-allocated data) are inherently shared among all threads of that process.
	- Local variables (on the stack) of a thread are private to that thread and not directly accessible to other threads.
	- Threads can be synchronized, access global memory areas, and variables of other threads.
	- Threads have a private address space, and communication between threads requires inter-process communication (IPC) facilitated by the operating system.
	- Threads are executed sequentially on the CPU, but hardware parallelism can be an exception.


A **Task Control Block (TCB)**, also referred to as a **Process Control Block (PCB)** is the generic term used in os theory to refer to the data structure that stores all the information about a process. A TCB typically contains the following information:

1. **Process Identifier (PID)**: A unique identifier for the process.
2. **Process State**: The current state of the process (e.g., running, waiting, blocked).
3. **Program Counter**: The address of the next instruction to be executed.
4. **CPU Registers**: Includes general-purpose registers, stack pointers, and program counters.
5. **Memory Management Information**: Information about the memory allocated to the process, including base and limit registers or page tables.
6. **Accounting Information**: Includes process execution times, user and kernel mode times, and other performance metrics.
7. **I/O Status Information**: Information about I/O devices allocated to the process, open file descriptors, etc.
8. **Scheduling Information**: Priority of the process, scheduling queue pointers, and other scheduling-related data.
9. **Other Information**: Security credentials, pointers to the process's parent process, signal handling information, etc.​​​​.

In the context of Linux, the PCB is implemented as the `task_struct`: 

|                          `task_struct` attributes                           |
| :-------------------------------------------------------------------------: |
| State {R (Running), I (Interruptible), U (Uninterruptible), D (Disk Sleep)} |
|                              PID (Process ID)                               |
|                          PPID (Parent Process ID)                           |
|                           mm (Memory Management)                            |
|                           fs (Filesystem Context)                           |
|                           files (Open Files List)                           |
|                          signal (Signal Handlers)                           |
|                 thread_struct (Processor Specific Context)                  |


When a context switch occurs (i.e., the CPU switches from executing one thread to another), the kernel uses the information in the `task_struct` to save the state of the current thread and restore the state of the next thread to run.
This is the state machine of a process: 

![](images/0fbb3bc5f8d6d9c9bf62a20044ee6bc5.png)


Linux treats all threads as standard processes. Each thread has a unique `task_struct` and appears to the kernel as a normal process, threads just happen to share resources, such as address space with other processes.

**System calls** and exception handlers are well-defined **interfaces** into the **kernel**. A process can begin executing in kernel space only through one of these interfaces, all access to the kernel is through these interfaces.

`fork()` is a system call in Linux used to create a new process. It duplicates the current process, known as the parent, to create a child process. The new `task_struct` of the child process is a copy of the parent’s, with differences in:

- **PID (Process ID)**: Unique identifier for the new process.
- **PPID (Parent Process ID)**: Set to the PID of the parent process.
- **Resources**: Some resources are duplicated or shared under certain conditions.

### System Initialization with systemd

**Systemd** is a system and service manager for Linux, operating as PID 1:

- It initializes the system, manages services, mounts HDDs, and handles clean-up.
- Replaces traditional init systems like SystemV with a more efficient and unified approach.
- Configuration files in declarative language called "unit files"
- Unit files are plain text, INI-style, encoding information about services, sockets, devices, mount points, automount points, etc.

**Systemctl** is command-line tool for querying and controlling the systemd system and service manager:

- Start, stop, and query the status of services.
- Manage system resources and services effectively.

### Task scheduling

The scheduler is responsible for determining the order in which tasks are executed ("task" can have different meanings and can be used interchangeably with "thread" in these notes).
A simplified view of task states is:

![](images/7d57cb3b57b34b223ff05125e8185379.png)

Actually the reality in Linux is much more complex and there are a lot of possible things that can happen to the process like errors, special states etc. 
