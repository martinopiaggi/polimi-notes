
# CUDA libraries 

CUDA offers GPU-accelerated libraries to optimize performance and enhance software productivity.
Libraries include:
        - **Math Libraries**: cuBLAS, cuFFT, CUDA Math Library, etc.
        - **Parallel Algorithm Libraries**: Thrust
        - **Image and Video Libraries**: nvJPEG, Video Codec SDK
        - **Communication Libraries**: NVSHMEM, NCCL
        - **Deep Learning Libraries**: cuDNN, TensorRT
        - **Partner Libraries**: OpenCV, Ffmpeg, ArrayFire


Detailed list and more info: [NVIDIA GPU-accelerated libraries](https://developer.nvidia.com/gpu-accelerated-libraries)

**Common Library Workflow**:

1. Create a library-specific handle.
2. Allocate device memory for inputs/outputs.
3. Format inputs to library-specific formats.
4. Execute the library function.
5. Retrieve outputs and convert them if necessary.
6. Release CUDA resources and continue with the application.

