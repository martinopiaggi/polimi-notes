# Read copy update locks 

Goal: low latency reads to shared data that is frequently read and nonfrequently written. Reads proceed without first acquiring a lock, by delaying the completion of a write.

## Readers:

- Avoid locks!
- Should tolerate concurrent writes (might be multiple concurrent versions)
- Might see old version for a limited time


## Writers:

- Create a new version (copy) of data structure
- Publish new version with a single atomic instruction

$\mathrm{RCU}$ is a synchronization mechanism that avoids the use of lock primitives while multiple threads concurrently read and update elements.

Whenever a thread is inserting or deleting elements of data structures in shared memory, all readers are guaranteed to see and traverse either the older or the new structure, therefore avoiding inconsistencies.

It is used when performance of reads is crucial and is an example of space-time trade-off, enabling fast operations at the cost of more space. This makes all readers proceed as if there were no synchronization involved, hence they will be fast, but also making updates more difficult.

Comparison with the Read-Write spin locks:

- Pros: avoid deadlocks and makes read lock acquisition a breeze.
- Cons: might delay the completion of destructive operations.


## Queued Spinlocks (MCS locks)

In symmetric multi-processor systems, every attempt to acquire a lock requires moving the cache line containing that lock to the local CPU. For contended locks, this cache-line bouncing can hurt performance significantly.

The queued spinlocks allow each processor to spin while checking its own memory location. The basic principle of a queue-based spinlock can best be understood by studying a classic queue-based spinlock implementation called the MCS lock.

The basic idea of the MCS lock is that a thread spins on a local variable and each processor in the system has its own copy of this variable. In other words this concept is built on top of the per-cpu variables concept in the Linux kernel.

When the first thread wants to acquire a lock, it registers itself in the queue. In other words, it will be added to the special queue and will acquire lock, because it is free for now. When the second thread wants to acquire the same lock before the first thread release it, this thread adds its own copy of the lock variable into this queue. In this case the first thread will contain a next field which will point to the second thread. From this moment, the second thread will wait until the first thread release its lock and notify next thread about this event. The first thread will be deleted from the queue and the second thread will be owner of a lock.

## Sleeping locks, Futex in User-space Concurrency

Futex (fast, user-level lock) Use cases: these are primarily used by posix threads to implement fast pthread mutex locks in user space.

## Goals:

- Avoid system calls, if possible, as system calls typically consume several hundred instructions
- Avoid unnecessary context switches
- Avoid thundering herd problems (all processes would be woken up to recheck their futexes only to have all of them requeued back again into another futex wait queue)


## Uncontended case behavior:

- Atomic operations are used to change the state of the futex in the uncontended case without the overhead of a syscall.
- Requires a shared state in user space accessible to all participating processes/task (a 32-bit integer) and user-space atomic operations


## Contended case behavior:

- The kernel is invoked to put tasks to sleep and wake them up.
- A kernel object is required, that has waiting queues associated with it.


## User-space event-based concurrency

Different style of concurrent programming used in GUI-based applications, internet servers. Useful when the state of each activity evolves with external events, not internal CPU instructions.

Idea: wait for something (i.e., an "event") to occur (or "posted" by an external entity). Check what type of event arrived, identify which activity it belongs and do the small amount of work it requires (which may include issuing I/O requests, or scheduling other events for future handling, etc.).

When a handler processes an event, it is the only activity taking place in the system.

- No locking needed: can modify shared data without worrying about other activities.
- No context switch overhead, except for resuming each activity from where it stopped.
- No uncontrollable pre-emption.
- A single stack can be used.

Use either select or poll APIs, they check whether there is any incoming network I/O that should be processed. They examine the first I/O descriptor sets checking if there is data ready to be read, or if the write buffer has still space, or if there is an exceptional condition pending. When context switches between threads are high overhead this is better than having multiple threads waiting for each single descriptor.

## Priority Inversion

Priority inversion is a scenario in scheduling in which a low priority task indirectly takes place of an higher priority task, effectively inverting the assigned priorities of the tasks. This violates the priority model that high priority tasks can only be prevented from running by even higher priority tasks. Inversion occurs when there is a resource contention with a low priority task that is then preempted by a medium priority task.

There is no foolproof method to predict the situation. There are however many existing solutions, of which the most common ones are:

- Disabling all interrupts to protect critical sections
- Priority ceiling protocol:

With priority ceiling protocol, the shared mutex process (that runs the operating system code) has a characteristic (high) priority of its own, which is assigned to the task locking the mutex. This works well, provided the other high priority tasks that tries to access the mutex does not have a priority higher than the ceiling priority.

- Priority inheritance:

Under the policy of priority inheritance, whenever a high priority task has to wait for some resource shared with an executing low priority task, the low priority task is temporarily assigned the priority of the highest waiting priority task for the duration of its own use of the shared resource, thus keeping medium priority tasks from pre-empting the (originally) low priority task, and thereby affecting the waiting high priority task as well. Once the resource is released, the low priority task continues at its original priority level.

- Avoid blocking:

Because priority inversion involves a low-priority task blocking a highpriority task, one way to avoid priority inversion is to avoid blocking, for example by using non-blocking algorithms such as read-copy-update.

## Executable Program Lifecycle

On UNIX systems, a loader program loads the contents of the executable file into memory. The loader program reads all code and initialized data into memory: the source code is put in read-only areas and initialized data ( $r / w)$ into address space.

In practice, lots of optimization might happen:

- Uninitialized data (bss) does not need to be read from memory.
- Demand-paging is used: loads pages only when it is used (lazy approach).
- Processes that share parts (libraries) or processes that are a copy of the same program will share code.

Definitions of BSS: block starting symbol (abbreviated to .bss or bss) is the portion of an object file, executable, or assembly language code that contains statically allocated variables that are declared but have not been assigned a value yet. It is often referred to as the "bss section". The BSS section does not contain bits relevant to the program; they will be initialized by the loader.

![](https://cdn.mathpix.com/cropped/2023_09_16_7a46320e788b77c73055g-04.jpg?height=577&width=948&top_left_y=1048&top_left_x=585)

## Sections of a compiled program

The differentiation between different purposes of a program part is represented by sections. We categorize them as static or dynamic depending on their actual behavior once in memory:

- Heap: allocated and laid out at runtime by malloc
- Namespace constructed dynamically, managed by programmer (names are stored in pointers, and organized using data structures).
- The compiler and linker are not involved other than saying where it can start.
- Stack: allocated at runtime with function calls, laid out by compiler
- Names are relative to a stack (or frame) pointer.
- Managed by compiler (allocated on procedure entry, freed on exit)
- Global data/code: allocated by compiler, laid out by linker
- Compiler creates them and names them with symbolic references.
- Linker lays them out and translates references.
- Mem-mapped regions: managed by programmer or linker
- Some programs directly call mmap. Dynamic linker uses these regions as well. Sections are intervals of addresses, allocated by the compiler and laid out in memory by the linker, that decides which virtual addresses to assign.

The ELF format is the way in which a compiled program is represented on disk and supports the loading process. The ELF acronym stands for Executable and Linkable Format. Every ELF file starts with a header and has several tables in it:

- program headers: describe common properties of section groups, also called segments.
- section headers: properties of each section.

A segment is a group of contiguous sections characterized by common properties. Each segment is represented in one program header, created by the linker. Each segment is specified by where it must be positioned, i.e. its offset within the file.

![](https://cdn.mathpix.com/cropped/2023_09_16_7a46320e788b77c73055g-05.jpg?height=608&width=571&top_left_y=901&top_left_x=731)

## Assembler create executables

Executables are built from object files by the linker. The object files are generated by the assembler. The compiled object files are built this way:

- Has sections but not segments
- Addresses of sections not yet decided, since they are decided by the linker
- It is called a relocatable file
- Contains additional sections to help the linker:
- Symbol table
- Relocation table

The assembler does not know where data/code should be placed in the address space. It also assumes each section starts at zero.

The assembler creates:

- a symbol table that holds the name and the offset of each created object
- a relocation table to instruct the linker where to fix the references; for this reason, it is called a relocatable file.

All the references in the object file must be rewritten once their value is known (after the linking process).

![](https://cdn.mathpix.com/cropped/2023_09_16_7a46320e788b77c73055g-06.jpg?height=351&width=1468&top_left_y=287&top_left_x=297)

The symbol table contains the symbols used in the source with their value being its address.

- If symbols are not defined in a section of an object file (they are specified elsewhere), then their value must be resolved during the linking process.
- The symbol type can be: object (for variables), function, file, section etc... or just undefined.

The relocation table tells the linker where code must be fixed in order to rewrite the references. Once the virtual addresses are known, the null immediate placeholders will be substituted by the linker.

## Linker

The linker operates in two phases:

## Phase 1:

- Coalesces all segments with the same name from all files.
- Determines the size of each segment and the resulting addresses, that specify where to place each object.
- Stores all global definitions in a global symbol table that maps the definitions to their final virtual addresses.


## Phase 2

- Ensures that each symbol has exactly one definition
- For each relocation, the linker looks up the referenced symbol virtual address in the symbol table and fixes all the references to reflect the addresses of the referenced symbols

The Linker uses special instructions in a script to build ELF files.

- It specifies the virtual memory address (VMA) of each section in the output file.
- It might also define symbols used within the program itself.
- The linker can also be instructed to substitute an instruction or a set of instructions with others in order to reduce the final code size. This technique is called relaxation.
- The dot character (.) is a running counter used to reference to the current virtual address.
- The asterisk symbol $(*)$ is used to collect all sections of the indicated type from input files.


## Dynamic shared libraries

## Problem:

- Everyone uses links to standard libraries
- These libraries consume code space in every executable

Idea for reducing the size of each executable:

- Have a single copy on disk of the library, represented with Shared objects (.so)
- Map, based on demand, the library in the address space of the requesting process
- Shared objects are just ELF files.

The ELF structure must point to the data to resolve the link at run-time. A global offset table contains the actual offset to jump but this is dynamically resolved on demand. Shared libraries are built to reside everywhere in memory no matter how many processes are using them.

## Programmable interrupt controller

![](https://cdn.mathpix.com/cropped/2023_09_16_7a46320e788b77c73055g-08.jpg?height=454&width=1552&top_left_y=390&top_left_x=250)

## Normal PIC --> Single CPU:

- Allowed to use only 1 CPU interrupt pin to manage 8 interrupts.
- Could be used in a chain (Master/slave) of two to increase IRQs to 15.
- Monitors the IRQ lines, checking for raised signals.


## APIC --> Multiple CPUs:

- 24 interrupts for all PCI devices, while XAPIC can handle 256 interrupts.
- Acts as a router with respect to the local APICs and can be programmed.
- It can deliver interrupts to multiple cores in a round robin fashion.
- Allows to generate inter-processor interrupts to enable a sort of message passing between CPUs.
- Note that interrupts might arrive unordered with respect to writes done by a device through DMA (known as consistency problem)


## MSI --> PCle lane used:

- Remove physical wires from devices. Uses PCle bus write messages instead
- Each device can produce up to 32 interrupt types and these are synchronous with respect to data read or written to memory.


## Inter-processor interrupts

An inter-processor interrupt (IPI) is not delivered through an IRQ line, but directly as a message on the bus that connects the local APIC of all CPUs. Linux makes use of three kinds of inter-processor interrupts:

- CALL_FUNCTION_VECTOR: Sent to all cpus except for the sender, forcing those cpus to run a function passed by the sender. For example, it can force all cpus to stop.
- RESCHEDULE_VECTOR: Force a cpu to reschedule its task.
- INVALIDATE_TLB_VECTOR: Sent to all cpus but the sender, forcing them to invalidate their Translation Lookaside Buffers.


## PCI (Peripheral Component Interconnect)

Stands for Peripheral Component Interconnect. It is a complete set of specifications defining how different parts of a computer should interact between them.

- Supports auto-detection of interface boards
- You plug in a jumper-less PCI device, and it is automatically configured at boot time.
- The overall layout of a $\mathrm{PCl}$ system is a tree where each bus is connected to an upper-layer bus, up to bus $\mathrm{O}$ at the root of the tree.

Each $\mathrm{PCl}$ peripheral is identified by a bus number, a device number, and a function number (a device might have multiple functions).

$\mathrm{PCl}$ configuration space is the underlying way that the $\mathrm{PCl}$ and $\mathrm{PCl}-$ Express perform auto configuration of the cards inserted into their bus.

At system boot, the firmware performs configuration transactions with every $\mathrm{PCl}$ peripheral in order to allocate a safe place for memory mapped 10 or port 10 regions it offers. The results of this operation are stored into an array of data structures called Base Address Registers (BARs), whose size is 256 bytes for each device function.

## BIOS start-up

- $\quad$ CPU is reset at start-up
- The BIOS has access to I/O interrupts and memory
- Caches and MMU are disabled
- Copies itself into RAM for faster access
- Looks for video adapters that may need to load their own routines
- Involves testing devices peripherals and the RAM
- Initializes connected devices such as the video card
- It does a RAM consistency check.
- Boot configuration (containing the boot order) loaded from non-volatile memory (NVRAM).

The BIOS tries to identify the stage 1 bootloader loading a sector from disk and jumps to it. This sector is called the Master Boot Record (MBR). It contains:

- Executable code
- A partition table
- Entry information: checks whether a partition what type of partition it is and if it is bootable


## Bootloaders

## Stage 1 bootloader

- Sets up the stack and, switches to 32-bit protected mode.
- Grand Unified Boot Loader (GRUB) is the most used stage 1 bootloader.


## Stage 1.5 bootloader

- Has code that reads/writes a file system
- Allows to read the bootloader stage 2 from the file system itself


## Stage 2 bootloader

- $\quad$ The second stage bootloader reads a configuration file, e.g. to startup a boot selection menu
- The kernel initial image is loaded in memory using BIOS disk I/O services


## Unified Extensible Firmware Interface (UEFI)

- Replacement of BIOS
- Modular: you can extend it with drivers
- Runs on various platforms and written in C
- Takes control right after the system is powered on and loads the firmware settings into RAM from nvRAM
- Start-up files are stored in a dedicated partition
- Overcomes size limitations of bios: BIOS only supported 4 partitions per disk, each up to 2.2TB. UEFl's global partition table uses 64 bits for logical block addresses, allowing a maximum disk size of $2 \wedge 64$ sectors.
- Overcomes other limitations of bios
- Bios could not understand file systems.
- UEFI understands and loads from a FAT boot partition.
- UEFI understands Microsoft's FAT file systems. Apple's UEFI knows HFS+ in addition


## Embedded Platforms

On embedded platforms:

- You typically have a very simple BootROM instead of the BIOS or UEFI
- Might need to load the kernel from network for quick development cycle
- Myriads of second stage bootloaders available, like u-Boot

U-Boot is a mature and well-known bootloader. It supports:

- 12 CPU architectures
- Lots of peripheral drivers: UART, SPI, I2C, Ethernet, SD, USB
- Can mount many different file systems
- Allows for flexible scripting in a shell


## Platform configuration

When booting, the OS must know:

- Which devices are already present on the machine
- How interrupts are managed
- How many processors there are

Peripheral devices are managed with a specific standard, called $\mathrm{PCl}$. However, this interface is not enough to find and configure everything in a platform. Two standards have evolved to provide the kernel with all the data of a platform (including $\mathrm{PCl}$ data):

- ACPI (Advanced Configuration and Power Interface): used mainly on general purpose platforms (Intel x86).
- Device trees: used mainly on embedded platforms.

The differences between them are:

- ACPI's byte code allows the platform to encode hardware behavior, while device trees do not.
- ACPI defines a power management model, while device trees do not.
- Device trees have been a quick solution for running Linux on vertically integrated devices (where one company controls both the product and its component parts). They are not really used in the PC field where software and hardware components are controlled by different companies.


## Platform discoverability

Provides an open standard for operating systems to:

- discover and configure computer hardware components,
- perform power management: e.g. putting unused hardware components to sleep
- perform auto configuration: e.g. plug and play and perform status monitoring
- No need to include platform-specific code for every platform
- No need to ship a separate (binary) kernel for every platform Not all devices/info are found on PCI buses: for example, $\mathrm{PCl}, \mathrm{CPU}$ enumeration, interrupt controllers, USB, SATA, and host bridges.

The ACPI namespace should describe everything the OS might use unless there's another way for the OS to find it.

Kernel receives a platform description in terms of tables:

- Tables contain code and references to registers
- Code is executed with an interpreter on behalf of device drivers and the kernel itself
- Tables specify a graph of properties and methods called the namespace

The ACPI namespace is a hierarchical data structure describing the underlying hardware platform. Linux builds an internal ACPI device tree.

## Device tree

The device tree is a data structure describing hardware. Device trees have both a binary format for operating systems to use and a textual format for convenient editing and management.

## What is the device tree used for:

- Usually passed to OS to provide information about HW topology where it cannot be detected/probed
- Move the hardware description out of the kernel binary
- No hard-coded initialization functions needed
- A single kernel can run on more than one board
- Useful if your platform does not provide any way to enumerate devices (PCI). In this case, you must provide BARs in a device tree

The nodes in the data structure are organized in a hierarchy as a collection of property and value tokens. The hierarchy represents the view of the system from the perspective of the CPU.

Interrupt signals are expressed as links between nodes independent of the tree. The meaning of an interrupt specifier depends entirely on the binding for the interrupt controller device. Each interrupt controller can decide how many cells it need to uniquely define an interrupt input.
