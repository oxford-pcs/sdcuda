#pragma once

#include "ccomplex.cuh"

// CUDA declarations
const int nCUDABLOCKS = 32;
const int nCUDATHREADSPERBLOCK = 256;

void cudaConvolveKernelReal2D(Complex*, Complex*, long, double*, long);
void cudaFftShift2D(Complex*, Complex*, long);
void cudaIFftShift2D(Complex*, Complex*, long);
void cudaScale2D(Complex*, double, long);
void cudaSetComplexRealAsAmplitude(Complex*, long);