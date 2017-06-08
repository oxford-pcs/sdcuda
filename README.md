# sdcuda
Spectral deconvolution on a GPU

# VS Project Setup

New Project > NVIDIA > Cuda 8.0 Runtime

Build -> Configuration Manager -> Active Solution Platform -> x64

VC++ Directories -> 
  Include Directories: $(VC_IncludePath);$(WindowsSDK_IncludePath);C:\CCfits\CCfits;C:\cfitsio\cfitsio
  Library Directories: $(VC_LibraryPath_x64);$(WindowsSDK_LibraryPath_x64);C:\CCfits\CCfits.build\Release;C:\cfitsio\cfitsio.build\Release
  
C/C++ ->
  Additional Include Directories: C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0\common\inc;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include\;C:\CCfits\CCfits;C:\CCfits\CCfits\..;%(AdditionalIncludeDirectories)

Linker ->
  Additional Dependencies: cufft.lib;cudart_static.lib;kernel32.lib;user32.lib;gdi32.l
