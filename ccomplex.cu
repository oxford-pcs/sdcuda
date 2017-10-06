#include "ccomplex.cuh"

#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>

#include "ckernels.h"

// DEVICE FUNCTIONS

__device__ __host__ Complex cAdd(Complex a, Complex b) {
	/*
	Add the real and imaginary components of two complex numbers.
	*/
	Complex c;
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	return c;
}

__device__ __host__ Complex cConvolveKernelReal(int i, Complex* a, long dim1, double* kernel, long kernel_size) {
	/*
	Convolve the real component of array [a] with kernel [kernel].
	*/
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
	/*
	Get the amplitude of the complex number [a].
	*/
	double abs = sqrt(pow(a.x, 2) + pow(a.y, 2));
	return abs;
}

__device__ __host__ double cGetPhase(Complex a) {
	/*
	Get the phase of the complex number [a].
	*/
	double phase = atan2(a.y, a.x);
	return phase;
}

__device__ __host__ long cGet1DIndexFrom2DXY(long2 xy, long dim1) {
	/*
	Given a pair of coordinates [xy] and array x dimension [dim1], find the corresponding 1D index.
	*/
	long index = (xy.y*dim1) + xy.x;
	return index;
}

__device__ __host__ long2 cGet2DXYFrom1DIndex(long index, long dim1) {
	/*
	Given a 1D index [index] and array x dimension [dim1], find the corresponding pair of coordinates xy.
	*/
	long2 xy;
	xy.x = index % dim1;
	xy.y = (long)(index / dim1);
	return xy;
}

__device__ __host__ quadrant cGet2DQuadrantFrom1DIndex(long index, long dim1, long x_split, long y_split) {
	/*
	Given a 1D index [index] and array x dimension [dim1], find the quadrant in which the index would lie 
	in a 2D array given the x/y split positions [x_split, y_split].
	*/
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

__device__ __host__ Complex cScale(Complex a, double s) {
	Complex c;
	c.x = a.x * s;
	c.y = a.y * s;
	return c;
}

__device__ __host__ Complex cSub(Complex a, Complex b) {
	/*
	Subtract a complex number [a] from a complex number [b].
	*/
	Complex c;
	c.x = a.x - b.x;
	c.y = a.y - b.y;
	return c;
}

// GLOBAL FUNCTIONS

__global__ void cAdd2D(Complex* a, Complex* b, long size) {
	/*
	Add the numbers from complex array [a] with [size] elements to complex array [b] pointwise.
	*/
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	// this is required as one thread may need to do multiple 
	// computations, i.e. if numThreads < size
	for (int i = threadID; i < size; i += numThreads) {
		a[i] = cAdd(a[i], b[i]);
	}
}

__global__ void cConvolveKernelReal2D(Complex* a, Complex* b, long dim1, double* kernel, long kernel_size) {
	/*
	Convolve kernel [kernel] of dimension [kernel_size] with complex array [a] of x dimension [dim1] and store 
	the result in [b].
	*/
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	long size = dim1*dim1;
	// this is required as one thread may need to do multiple 
	// computations, i.e. if numThreads < size
	for (int i = threadID; i < size; i += numThreads) {
		long2 xy = cGet2DXYFrom1DIndex(i, dim1);
		long kernel_half_size = (kernel_size - 1) / 2;
		if (xy.x >= kernel_half_size && xy.x < dim1 - kernel_half_size && xy.y >= kernel_half_size && xy.y < dim1 - kernel_half_size) {
			b[i] = cConvolveKernelReal(i, a, dim1, kernel, kernel_size);
		}
	}
}

__global__ void cFftShift2D(Complex* a, Complex* b, long dim) {
	/*
	Perform an fftshift on a complex array [a] to yield [b]. This routine only handles arrays with dimensions of equal size [dim].
	*/
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
		switch (cGet2DQuadrantFrom1DIndex(i, dim, x_split, y_split)) {
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

__global__ void cIFftShift2D(Complex* a, Complex* b, long dim) {
	/*
	Perform an ifftshift on a complex array [a] to yield [b]. This routine only handles arrays with dimensions of equal size [dim].
	*/
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
		switch (cGet2DQuadrantFrom1DIndex(i, dim, x_split, y_split)) {
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

__global__ void cScale2D(Complex *a, double scale, long size) {
	/*
	Scale complex array [a] with [size] elements by [scale] pointwise.
	*/
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	// this is required as one thread may need to do multiple 
	// computations, i.e. if numThreads < size
	for (int i = threadID; i < size; i += numThreads) {
		a[i] = cScale(a[i], scale);
	}
}

__global__ void cSub2D(Complex* a, Complex* b, long size) {
	/* 
	Subtract array [b] with [size] elements from array [a] pointwise.
	*/
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	// this is required as one thread may need to do multiple 
	// computations, i.e. if numThreads < size
	for (int i = threadID; i < size; i += numThreads) {
		a[i] = cSub(a[i], b[i]);
	}
}

__global__ void cSetComplexRealAsAmplitude(Complex *a, long size) {
	/*
	Set the real component of array [a] with [size] elements to the amplitude and zero the  
	imaginary part.
	*/
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	// this is required as one thread may need to do multiple 
	// computations, i.e. if numThreads < size
	for (int i = threadID; i < size; i += numThreads) {
		a[i].x = cGetAmplitude(a[i]);
		a[i].y = 0;
	}
}
