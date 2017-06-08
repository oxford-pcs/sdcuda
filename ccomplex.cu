#include "ccomplex.cuh"

#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>

#include "ckernels.h"

__device__ __host__ Complex cAdd(Complex a, Complex b)
{
	Complex c;
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	return c;
}

__device__ __host__ Complex cConvolveKernel(int i, Complex* a, long dim1, double* kernel, long kernel_size) {
	long kernel_half_size = (kernel_size - 1) / 2;
	Complex new_value;
	new_value.x = 0;
	new_value.y = 0;
	for (int kj = 0; kj < kernel_size; kj++) {
		for (int ki = 0; ki < kernel_size; ki++) {
			new_value.x += a[i + ((ki - kernel_half_size) + (kj - kernel_half_size)*dim1)].x * kernel[(kj*kernel_size) + ki];
		}
	}
	return new_value;
}

__device__ __host__ double cGetAmplitude(Complex a) {
	double abs = sqrt(pow(a.x, 2) + pow(a.y, 2));
	return abs;
}

__device__ __host__ double cGetPhase(Complex a) {
	double phase = atan2(a.y, a.x);
	return phase;
}

__device__ __host__ long cGet1DIndexFrom2DXY(long2 xy, long dim1) {
	long index = (xy.y*dim1) + xy.x;
	return index;
}

__device__ __host__ long2 cGet2DXYFrom1DIndex(long index, long dim1) {
	long2 xy;
	xy.x = index % dim1;
	xy.y = (long)(index / dim1);
	return xy;
}

__device__ __host__ quadrant cGet2DQuadrantFrom1DIndex(long index, long dim1, long dim2, long x_split, long y_split) {
	long2 xy = cGet2DXYFrom1DIndex(index, dim1);
	if (xy.x < x_split) {
		if (xy.y < y_split) {
			return Q1;
		}
		else {
			return Q3;
		}
	}
	else {
		if (xy.y < y_split) {
			return Q2;
		}
		else {
			return Q4;
		}
	}
}

__device__ __host__ Complex cScale(Complex a, double s)
{
	Complex c;
	c.x = a.x * s;
	c.y = a.y * s;
	return c;
}

__device__ __host__ Complex cSub(Complex a, Complex b)
{
	Complex c;
	c.x = a.x - b.x;
	c.y = a.y - b.y;
	return c;
}

__global__ void cAddPointwise2D(Complex* a, Complex* b, long size)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	// this is required as one thread may need to do multiple 
	// computations, i.e. if numThreads < size
	for (int i = threadID; i < size; i += numThreads) {
		a[i] = cAdd(a[i], b[i]);
	}
}

__global__ void cConvolveKernelPointwise(Complex* a, Complex* b, long dim1, double* kernel, long kernel_size) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	long size = dim1*dim1;

	// this is required as one thread may need to do multiple 
	// computations, i.e. if numThreads < size
	for (int i = threadID; i < size; i += numThreads) {
		long2 xy = cGet2DXYFrom1DIndex(i, dim1);
		long kernel_half_size = (kernel_size - 1) / 2;
		if (xy.x >= kernel_half_size && xy.x < dim1 - kernel_half_size && xy.y >= kernel_half_size && xy.y < dim1 - kernel_half_size) {
			b[i] = cConvolveKernel(i, a, dim1, kernel, kernel_size);
		}
	}
}

__global__ void cFftShift2D(Complex* a, Complex* b, long dim)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	long size = dim*dim;
	long Q1_offset, Q2_offset, Q3_offset, Q4_offset;
	if (dim % 2 == 0) {
		Q1_offset = (dim*((dim) / 2.)) + ((dim) / 2);
		Q2_offset = (dim*((dim) / 2)) - ((dim) / 2);
		Q3_offset = -(dim*((dim) / 2)) + ((dim) / 2);
		Q4_offset = -(dim*((dim) / 2)) - ((dim) / 2);
	}
	else {
		Q1_offset = (dim*ceil(dim / 2.)) + ceil(dim / 2.);
		Q2_offset = (dim*ceil(dim / 2.)) - floor(dim / 2.);
		Q3_offset = -(dim*floor(dim / 2.)) + ceil(dim / 2.);
		Q4_offset = -(dim*floor(dim / 2.)) - floor(dim / 2.);
	}

	long x_split = floor(dim / 2.);
	long y_split = floor(dim / 2.);

	for (int i = threadID; i < size; i += numThreads) {
		switch (cGet2DQuadrantFrom1DIndex(i, dim, dim, x_split, y_split)) {
		case Q1:
			b[i] = a[i + Q1_offset];
			break;
		case Q2:
			b[i] = a[i + Q2_offset];
			break;
		case Q3:
			b[i] = a[i + Q3_offset];
			break;
		case Q4:
			b[i] = a[i + Q4_offset];
			break;
		}
	}
}

__global__ void cIFftShift2D(Complex* a, Complex* b, long dim)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	long size = dim*dim;
	long Q1_offset, Q2_offset, Q3_offset, Q4_offset;
	if (dim % 2 == 0) {
		Q1_offset = (dim*((dim) / 2)) + ((dim) / 2);
		Q2_offset = (dim*((dim) / 2)) - ((dim) / 2);
		Q3_offset = -(dim*((dim) / 2)) + ((dim) / 2);
		Q4_offset = -(dim*((dim) / 2)) - ((dim) / 2);
	}
	else {
		Q1_offset = (dim*floor(dim / 2.)) + floor(dim / 2.);
		Q2_offset = (dim*floor(dim / 2.)) - ceil(dim / 2.);
		Q3_offset = -(dim*ceil(dim / 2.)) + floor(dim / 2.);
		Q4_offset = -(dim*ceil(dim / 2.)) - ceil(dim / 2.);
	}
	long x_split = ceil(dim / 2.);
	long y_split = ceil(dim / 2.);

	for (int i = threadID; i < size; i += numThreads) {
		switch (cGet2DQuadrantFrom1DIndex(i, dim, dim, x_split, y_split)) {
		case Q1:
			b[i] = a[i + Q1_offset];
			break;
		case Q2:
			b[i] = a[i + Q2_offset];
			break;
		case Q3:
			b[i] = a[i + Q3_offset];
			break;
		case Q4:
			b[i] = a[i + Q4_offset];
			break;
		}
	}
}

__global__ void cScalePointwise(Complex *a, double scale, long size)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	// this is required as one thread may need to do multiple 
	// computations, i.e. if numThreads < size
	for (int i = threadID; i < size; i += numThreads) {
		a[i] = cScale(a[i], scale);
	}
}

__global__ void cSubPointwise2D(Complex* a, Complex* b, long size)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	// this is required as one thread may need to do multiple 
	// computations, i.e. if numThreads < size
	for (int i = threadID; i < size; i += numThreads) {
		a[i] = cSub(a[i], b[i]);
	}
}

__global__ void cSetComplexRealAsAmplitude(Complex *a, long size) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	// this is required as one thread may need to do multiple 
	// computations, i.e. if numThreads < size
	for (int i = threadID; i < size; i += numThreads) {
		a[i].x = cGetAmplitude(a[i]);
		a[i].y = 0;
	}
}

__global__ void cShift2D(Complex* a, Complex* b, long dim, long shift_x, long shift_y) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	long size = dim*dim;

	// this is required as one thread may need to do multiple 
	// computations, i.e. if numThreads < size
	for (int i = threadID; i < size; i += numThreads) {
		long2 xy = cGet2DXYFrom1DIndex(i, dim);
		xy.x += shift_x;
		xy.y += shift_y;
		long new_index;
		if (xy.x < dim && xy.y < dim) {
			b[cGet1DIndexFrom2DXY(xy, dim)] = a[i];
		}
	}
}
