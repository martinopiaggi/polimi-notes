
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
