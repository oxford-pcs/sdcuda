#include "cmemory.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <valarray>

#include "ccomplex.cuh"

using std::valarray;

int memory::memcpydd(Complex* dst, Complex* src, long size) {
	/*
	Copy block of memory of size [size] from device location [src] to device location [dst].
	*/
	cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to copy memory to device\n");
		return 1;
	}
	return 0;
}

int memory::memcpydh(Complex* dst, Complex* src, long size) {
	/*
	Copy block of memory of size [size] from device location [src] to host location [dst].
	*/
	cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to copy memory to host\n");
		return 1;
	}
	return 0;
}

int memory::memcpyhd(Complex* dst, Complex* src, long size) {
	/*
	Copy block of memory of size [size] from host location [src] to device location [dst].
	*/
	cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to copy memory to device\n");
		return 1;
	}
	return 0;
}

int memory::memcpyhh(Complex* dst, Complex* src, long size) {
	/*
	Copy block of memory of size [size] from host location [src] to host location [dst].
	*/
	cudaMemcpy(dst, src, size, cudaMemcpyHostToHost);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to copy memory to host\n");
		return 1;
	}
	return 0;
}


int hmemory::free(Complex* data) {
	/*
	Free host memory block.
	*/
	if (data != NULL) {
		cudaFreeHost(data);
		if (cudaGetLastError() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to free memory on host\n");
		}
	}
	return 0;
}

Complex* hmemory::malloc(long size, bool zero_initialise) {
	/*
	Allocate a piece of memory on the host of size [size].
	*/
	Complex* data = NULL;
	cudaMallocHost(&(data), size);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to allocate\n");
	}
	if (zero_initialise) {
		memset(data, 0, size);
	}
	return data;
}


int dmemory::free(Complex* data) {
	/*
	Free device memory block.
	*/
	if (data != NULL) {
		cudaFree(data);
		if (cudaGetLastError() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to free memory on device\n");
		}
	}
	return 0;
}

Complex* dmemory::malloc(long size, bool zero_initialise) {
	/*
	Allocate a piece of memory on the device of size [size].
	*/
	Complex* data = NULL;
	cudaMalloc((void**)&(data), size);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to allocate\n");
	}
	if (zero_initialise) {
		cudaMemset(data, 0, size);
		if (cudaGetLastError() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to memset\n");
		}
	}
	return data;
}
