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

Complex* hmemory::realloc(Complex* old_data, long new_size, long old_size, bool zero_initialise_if_grow) {
	/*
	Reallocate a piece of memory of size [old_size] on the host of size [new_size].
	*/
	Complex* new_data = NULL;
	cudaMallocHost((void**)&(new_data), new_size);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to allocate\n");
	}
	if (new_size > old_size && zero_initialise_if_grow) {
		memset(new_data, 0, new_size);
		if (cudaGetLastError() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to memset\n");
		}
	}
	dmemory::memcpyhh(new_data, old_data, new_size);
	free(old_data);

	return new_data;
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

Complex* dmemory::realloc(Complex* old_data, long new_size, long old_size, bool zero_initialise_if_grow) {
	/*
	Reallocate a piece of memory of size [old_size] on the device of size [new_size].
	*/
	Complex* new_data = NULL;
	cudaMalloc((void**)&(new_data), new_size);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to allocate\n");
	}
	if (new_size > old_size && zero_initialise_if_grow) {
		cudaMemset(new_data, 0, new_size);
		if (cudaGetLastError() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to memset\n");
		}
	}
	dmemory::memcpydd(new_data, old_data, new_size);
	free(old_data);

	return new_data;
}
