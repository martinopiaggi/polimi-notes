# Booting

When booting, the operating system (OS) needs to be aware of several things: 

1. The devices that are already present on the machine.
2. How interrupts are managed.
3. The number of processors available.
4. Peripheral device standards such as PCI are not sufficient to identify and configure everything on a platform. 

To address these needs, two standards have been developed to provide the kernel with all the necessary platform data (including PCI data): 

- **Advanced Configuration and Power Interface** (ACPI): This standard is primarily used on general-purpose platforms, particularly those utilizing Intel processors. 
- **Device trees**: This standard is mainly used on embedded platforms. Device trees aid in discoverability and provide information about the platform's configuration.

## ACPI 

ACPI helps with discoverability, power management, and thermal management. It was developed by Intel, Microsoft, and Toshiba in 1996. ACPI aims to improve discoverability, power management, and thermal management on these platforms. The key benefits of ACPI are:
	- It provides an open standard for operating systems to discover and configure computer hardware components.
	- It enables power management features such as putting unused hardware components to sleep.
	- It supports auto-configuration, including Plug and Play and hot swapping.
	- It facilitates status monitoring.

By implementing ACPI, there is no need to include platform-specific code for every platform. Additionally, there is no need to ship a separate binary kernel for each platform, streamlining the development and deployment processes.

### ACPI Power Management

Power management refers to the various tools and techniques used to consume the minimum amount of power necessary based on the system state, configuration, and use case.

**Goals**:

1. Improve battery lifetime by reducing energy consumption. Energy is the integration of power over time.
2. Enhance reliability by managing temperature. Higher temperatures, which are proportional to power and operational time, can decrease component reliability.

**Power Management Problem**:
Considerations:
The primary source of power consumption is related to the frequency and voltage of each system device.

Key Questions:
1. What voltage should be used for each device?
2. What frequency should be used for each device?
3. How does switching between different frequencies and voltages impact the system?

Addressing the power management problem in an industrial setting is challenging to do mathematically. However, it has been simplified by using **device states**.
ACPI models device states by designing devices to operate within specific voltage and frequency ranges. Going beyond these ranges can lead to malfunctions, excessive heat, or permanent damage. Switching voltage and frequency is not a simple task and consumes time, energy, and computational resources. If not managed properly, the energy saved by reducing power could be offset by the energy spent managing transitions.

Power management is complex due to the interdependence of multiple variables. To simplify this, devices are designed to operate in predefined states with specific voltage and frequency settings. ACPI exposes all available states of a device to the operating system, making power state management easier.

ACPI also provides mechanisms for putting the entire computer in and out of system sleeping states. ACPI tables describe motherboard devices, their power states, power planes, and controls for putting devices into different power states.

This allows the OS to put devices into low-power states based on application usage. The specific power management policy is left to the OSPM.

ACPI defines a hierarchical state machine to describe the power state of a system. There are five different power state types: G-type, S-type, C-type, P-type, and D-type.

- G-type states group together system states (S-type) and provide a coarse-grained description of the system's behavior. They can indicate whether the system is working, appearing off but able to wake up on specific events, or completely off.
- S-type states, also known as system or sleep states, describe how the computer implements the corresponding G-type state. For example, the G1 state can be implemented in different ways, such as halting the CPU while keeping the RAM on, or completely powering off the CPU and RAM and moving memory content to disk.
- C-type states, specific to the CPU (CPU State), refer to the power management capabilities of the CPU. These states allow the CPU to reduce its clock speed (C0) or even shut down completely when idle to save energy (C1-C3). C-states can only be entered in an S0 configuration (working state) and are usually invoked through a specific CPU instruction, such as MWAIT on Intel processors.
- D-type states (Device State) are the equivalent of C-states but for peripheral devices. They represent the power management capabilities of the devices.
- P-type states (Performance State) are a power management feature that allows the CPU to adjust its clock speed and voltage based on workload demands. P-states only make sense in the C0 configuration (working state).

These different ACPI power state types provide a structured way to manage the power consumption of a computer system, allowing for efficient power management and low-power operation when appropriate.

In the ideal case, these two situations are equal, but in reality, idle time prediction is not always accurate. When dealing with interactive systems or real-time systems, it is better to use voltage and frequency scheduling instead of relying on idle time prediction.

C3 may provide better advantages in terms of thermal regime, but it has higher wakeup times. On the other hand, for interactive systems or real-time systems, it is more beneficial to use voltage and frequency scheduling.

When an operating system detects that a system is not being used or is idle, it needs to make decisions regarding power management. This includes determining whether to keep the CPU active, move it to an idle state, or transition it to a low power state. For example, if we anticipate that the system will be idle for 4 seconds, we need to decide whether to push the frequency to finish the job quickly or slow it down. However, in reality, idle time prediction is not always accurate, and unforeseen events may occur if the system is interactive.

The power management stack is divided into two parts: the ACPI daemon and the OSPM. When the scheduler identifies that more performance is required (a P-state change), it communicates this desired change to the CPUFreq module in OSPM. CPUFreq then interacts with ACPI to set the new P-state. ACPI triggers this change by modifying the Performance Control Machine register. The ACPI daemon listens for power events and manages system-wide power states S and G.

For example, when a laptop lid is closed, an OS-visible interrupt called a "system control interrupt" (SCI) is generated. The ACPI driver in the kernel exposes this event to user space through the sysfs interface. The ACPI daemon, which monitors sysfs, can take appropriate actions, such as putting the laptop into sleep mode (S3 state), after notifying all applications of the impending transition. The actual transition is initiated by writing to /sys/power/state. This prompts the OS to notify drivers and freeze applications. A call is then made to the BIOS to enter this state. ACPI specifies a number of kernel-controlled power management control (PM) registers for enabling/disabling SCI interrupts (SCI_EN) and entering a specific type of system sleep S-state through (SLP_EN, SLP_TYP), but only after the kernel has successfully saved state.


## Device trees



## Extra


During machine boot-up, the firmware (BIOS/EFI) is loaded into main memory and performs necessary checks before launching the bootloader. The bootloader mounts the necessary filesystem and loads the kernel image. In Linux, the init process is started at the end of the kernel boot. The init process has a PID of 1 and PPID of 0. It is the ancestor of all user-space processes and starts all enabled services during startup. The two common implementations of init are SysVinit and systemd.

When executing a program, the init process forks itself and uses the execve system call family to load the executable path and execute the commands inside the program.

```cpp
#include <unistd. h>
int execl(const char *path, const char *arg, - );
int execlp(const char *file, const char *arg, - );
int execle(const char *path, const char *arg, -, char * const envp[]);
int execv(const char *path, char * const argv[]);
int execvp(const char *file, char * const argv[]);
int execve(const char *path, char * const argv[], char * const envp[]);
```


By using one of these functions, init can effectively start new processes during system startup or when certain events occur. 

There are three versions of these functions. The first argument is the executable file on which we want to replace the forked space. The second argument is the list of arguments for the program, typically passed as command-line arguments.

The first three versions use a constant char pointer for the arguments, while the "-V" version requires an array of strings to be specified.

The "-p" versions only require specifying the name of the binary because it can be located in the path environment variable. This allows accessing variables and environment settings during system startup or runtime.

If we need to specify the entire path to a binary that is not in one of these common paths, we would use one of the non-"-p" versions.

