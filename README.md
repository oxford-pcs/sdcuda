# sdcuda
Perform spectral deconvolution on a CUDA enabled machine.

# Building

## VS Project

The build uses three external libraries: CCfits, CUDA and CULA. CULA is a proprietary library that is free to use for academic purposes. 

To build from scratch, first switch the active build to x64:

`Build -> Configuration Manager` and switch `Active Solution Platform` to `x64`. 

Then add the libraries and their headers to the relevant build directories:

In `Project -> sdcuda Properties -> Configuration Properties -> C/C++ -> General -> Additional Include Directories`, add the necessary include paths, e.g. 

`C:\cfitsio\cfitsio;C:\CCfits\CCfits\..;C:\CCfits\CCfits;C:\CULA\include;C:\ProgramData\NVIDIA Corporation\CUDA Samples\v7.5common\inc;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include\`

and similarly, add the library paths to `Project -> sdcuda Properties -> Configuration Properties -> Linker -> General -> Additional Library Directories`, e.g.

`$(CudaToolkitLibDir);C:\CCfits\CCfits.build\Release;C:\cfitsio\cfitsio.build\Release;C:\CULA\lib`

then add the specific libraries required to `Project -> sdcuda Properties -> Configuration Properties -> Linker -> Input -> Additional Dependencies`, e.g. 

`cublas.lib;cufft.lib;cudart_static.lib;kernel32.lib;user32.lib;gdi32.lib;winspool.lib;comdlg32.lib;advapi32.lib;shell32.lib;ole32.lib;oleaut32.lib;uuid.lib;odbc32.lib;odbccp32.lib;kernel32.lib;user32.lib;gdi32.lib;winspool.lib;shell32.lib;ole32.lib;oleaut32.lib;uuid.lib;comdlg32.lib;advapi32.lib;C:\CCfits\CCfits.build\Release\CCfits.lib;C:\cfitsio\cfitsio.build\Release\cfitsio.lib;C:\CULA\lib\culapack.lib;C:\CULA\lib\culapack_link.lib`

## Release

The CULA and CUDA .dlls need to be placed in the executable directory (cublas64\_XX, cudart64\_XX, cula_sparse, culapack, culapack_link).

## Configuration

The device tag defines attributes related to the CPU processing. The program's configuration parameters are stored in the distribution's `config.xml` file. 
There are three distinct sections under the XML root: `<host>`, `<device>` and `<process>`.

### `<host>`

A host item is encapsulated with a `<param>` tag, each of which should have `<name>`, `<value>` and `<description>` tags. The description tag is purely aesthetic. 
Recognised parameter name/values are:

- nCPUCORES (int) - The number of CPU cores to use in multiprocessing. Setting this parameter too high will move the bottleneck to CPU memory.


### `<device>`

The device tag defines attributes related to the GPU processing. A device item is encapsulated with a `<param>` tag, each of which should have `<name>`, `<value>` 
and `<description>` tags. The description tag is purely aesthetic. Recognised parameter names (and their expected value types) are:

- nCUDABLOCKS (int) - The number of CUDA blocks.
- nCUDATHREADSPERBLOCK (int) - The number of threads per CUDA block.

### `<process>`

The process tag defines the sequence of stages in the pipeline. Each process item is encapsulated with a `<stage>` tag, each of which should have a `<name>` tag. 
Recognised stage names can be found in the enum `process_stages` in `cprocess.h`. Stages will be actioned in the order they are placed in the file.

# Executing

The binary is called with the following command line parameters (with corresponding flags) required:

- The input FITS file path (-i)
- The simulation parameters file path (-p)
- The configuration file path (-c) 
- The output file path (-o)

e.g. `sdcuda -i "C:\Users\barnsley\Desktop\HARMONI-HC_data\in.fits" -p "C:\Users\barnsley\Desktop\HARMONI-HC_data\parameters.xml" -c "C:\Users\barnsley\Documents\Visual Studio 2013\Projects\sdcuda\config.xml" -o "C:\Users\barnsley\Desktop\HARMONI-HC_data\out.fits"`

# Architecture

The design of the program revolves around the `cube` class type and its two derived classes `hcube` and `dcube`. The prefixes of the two derived classes denote 
where the data will physically reside, either on the host (h) or on the device (d). 

Although the program takes a 4D file as input, with separation integrations for each rotator position (as required by ADI), a `cube` instance is defined as a 
datacube in the traditional sense, with two spatial axes and one spectral. Each cube houses the location of its data in a vector container of `spslice` type, 
with derived classes `hspslice` and `dspslice`. Note that data on the device cannot be read outside of a device function and must be copied to the host - not 
doing so will yield a segfault.

To decompose the 4D input, each integration is put in to a queue with maximum concurrent integration reductions defined by the `host` parameter `nCPUCORES`. The 
corresponding cube for the integration is then constructed before processing.

# Common How-Tos

## Adding a Process 

To create a new process stage, the following sequence of events should be followed:

1. Add a unique process identifier to `process_stages` in `cprocess.h`. 
2. Add a new case to `process::step` in `cprocess.cpp`. This may require adding additional functions and prototypes to `cprocess.cpp` and `cprocess.h`.
3. Add the corresponding XML stage name -> process_stage mapping to `process_stages_mapping` in `cinput.h`.

## Adding a CUDA Device Call

If a call to a function to be performed on the device is required, it will be necessary to define a new device function. A device function is called via:

`cudacalls.cuh` function > \_global\_ `cdevice.cuh` function -> async \_device\_ `cdevice.cuh` function

The following sequence of events should be followed:

1. Add the function prototype to `cudacalls.cuh`. The first two parameters of the function call will always be nCUDABLOCKS and nCUDATHREADSPERBLOCK, so they 
should always have `int` type. 
2. Define the corresponding function call in `cudacalls.cu`.
3. Add a new `__global__` function prototype to `cdevice.cuh`.
4. Define the corresponding function call in `cdevice.cu`.
5. (optional) Add necessary `__device__` prototypes and defintions to `cdevice.cuh` and `cdevice.cu` respectively. These are the functions that will be called 
asynchronously for each thread on the GPU.
