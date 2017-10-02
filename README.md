# sdcuda
Spectral deconvolution on a GPU

# VS Project Setup

New Project > NVIDIA > Cuda 8.0 Runtime

Build -> Configuration Manager -> Active Solution Platform -> x64

VC++ Directories -> 
  Include Directories: $(VC_IncludePath);$(WindowsSDK_IncludePath);C:\CCfits\CCfits;C:\cfitsio\cfitsio;C:\CULA\include
  Library Directories: $(VC_LibraryPath_x64);$(WindowsSDK_LibraryPath_x64);C:\CCfits\CCfits.build\Release;C:\cfitsio\cfitsio.build\Release;C:\CULA\lib
  
C/C++ ->
  Additional Include Directories: C:\ProgramData\NVIDIA Corporation\CUDA Samples\v7.5common\inc;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include\;C:\CCfits\CCfits;C:\CCfits\CCfits\..;C:\CULA\include%(AdditionalIncludeDirectories)

Linker -> Input ->
  Additional Dependencies: cufft.lib;cudart_static.lib;kernel32.lib;user32.lib;gdi32.l;culapack.lib;culapack_link.lib

# Binary setup

CULA and CUDA .dlls need to be placed in the executable directory.
