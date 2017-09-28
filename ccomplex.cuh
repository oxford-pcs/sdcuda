#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

typedef double2 Complex;

enum quadrant {
	NONE,
	Q1,
	Q2,
	Q3,
	Q4
};

enum complex_part {
	REAL,
	IMAGINARY,
	AMPLITUDE,
	PHASE
};

__device__ __host__ Complex cAdd(Complex, double);
__device__ __host__ Complex cConvolveKernel(int, Complex*, long, double*, long);
__device__ __host__ double cGetAmplitude(Complex);
__device__ __host__ double cGetPhase(Complex);
__device__ __host__ long cGet1DIndexFrom2DXY(long2, long);
__device__ __host__ long2 cGet2DXYFrom1DIndex(long, long);
__device__ __host__ quadrant cGet2DQuadrantFrom1DIndex(long, long, long, long, long);
__device__ __host__ Complex cScale(Complex, double);
__global__ void cAddPointwise2D(Complex*, Complex*, long);
__global__ void cConvolveKernelPointwise(Complex*, Complex*, long, double*, long);
__global__ void cFftShift2D(Complex*, Complex*, long);
__global__ void cIFftShift2D(Complex*, Complex*, long);
__global__ void cScalePointwise(Complex*, double, long);
__global__ void cSetComplexRealAsAmplitude(Complex*, long);
__global__ void cShift2D(Complex*, Complex*, long, long, long);
__global__ void cSubPointwise2D(Complex*, Complex*, long);