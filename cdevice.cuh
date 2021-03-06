#pragma once

#define M_PI	3.14159265358979323846

#include <cuda_runtime.h>
#include <cufft.h>
#include <cula.h>

#include "ccomplex.h"

enum quadrant {
	NONE,
	Q1,
	Q2,
	Q3,
	Q4
};

__device__ __host__ Complex cAdd(Complex, Complex);
__device__ __host__ Complex cExp(Complex);
__device__ __host__ Complex cMultiply(Complex, Complex);
__device__ __host__ Complex cMultiply(Complex, double);
__device__ __host__ Complex cSub(Complex, Complex);

__device__ __host__ void cCompareArray(int**, int*, long, long, long);
__device__ __host__ double cGetAmplitude(Complex);
__device__ __host__ double cGetPhase(Complex);
__device__ __host__ void cGetSpaxelData(Complex**, Complex*, long, long);
__device__ __host__ long cGet1DIndexFrom2DXY(long2, long);
__device__ __host__ long2 cGet2DXYFrom1DIndex(long, long);
__device__ __host__ quadrant cGet2DQuadrantFrom1DIndex(long, long, long, long);
__device__ __host__ void cMakeBitmask(Complex**, int*, long, long);
__device__ __host__ void cPolySub(Complex*, Complex*, long, long);
__device__ __host__ Complex cTranslate(Complex, long, long, long2, double2);

__global__ void cAdd2D(Complex*, Complex*, long);
__global__ void cCompareArray2D(int**, int*, long, long, long);
__global__ void cDivideByRealComponent2D(Complex*, Complex*, Complex*, long);
__global__ void cGetSpaxelData2D(Complex**, Complex**, long, long);
__global__ void cFftShift2D(Complex*, Complex*, long);
__global__ void cIFftShift2D(Complex*, Complex*, long);
__global__ void cMakeBitmask2D(Complex**, int**, long, long);
__global__ void cMultiplyHadamard2D(Complex*, Complex*, Complex*, long);
__global__ void cScale2D(Complex*, double, long);
__global__ void cSetComplexRealAsAmplitude2D(Complex*, long);
__global__ void cSub2D(Complex*, Complex*, long);
__global__ void cPolySub2D(Complex**, int**, Complex**, long, int*, long, long);
__global__ void cTranslate2D(Complex*, double2, long);