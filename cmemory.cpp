#include "cmemory.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <valarray>

#include "ccomplex.cuh"
#include "errors.h"

using std::valarray;

int memory::memcpydd(Complex* dst, Complex* src, long size) {
	/*
	Copy block of memory of size [size] from device location [src] to device location [dst].
	*/
	cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
	if (cudaGetLastError() != cudaSuccess) {
		throw_error(CUDA_FAIL_MEMCPY_DD);
	}
	return 0;
}

int memory::memcpydh(Complex* dst, Complex* src, long size) {
	/*
	Copy block of memory of size [size] from device location [src] to host location [dst].
	*/
	cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
	if (cudaGetLastError() != cudaSuccess) {
		throw_error(CUDA_FAIL_MEMCPY_DH);
	}
	return 0;
}

int memory::memcpyhd(Complex* dst, Complex* src, long size) {
	/*
	Copy block of memory of size [size] from host location [src] to device location [dst].
	*/
	cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
	if (cudaGetLastError() != cudaSuccess) {
		throw_error(CUDA_FAIL_MEMCPY_HD);
	}
	return 0;
}

int memory::memcpyhh(Complex* dst, Complex* src, long size) {
	/*
	Copy block of memory of size [size] from host location [src] to host location [dst].
	*/
	cudaMemcpy(dst, src, size, cudaMemcpyHostToHost);
	if (cudaGetLastError() != cudaSuccess) {
		throw_error(CUDA_FAIL_MEMCPY_HH);
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
			throw_error(CUDA_FAIL_FREE_MEMORY_H);
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
		throw_error(CUDA_FAIL_ALLOCATE_MEMORY_H);
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
	if (new_size > old_size) {
		if (zero_initialise_if_grow) {
			new_data = hmemory::malloc(new_size, true);
		} else {
			new_data = hmemory::malloc(new_size, false);
		}
		hmemory::memcpyhh(new_data, old_data, old_size);
	} else {
		new_data = hmemory::malloc(new_size, false);
		hmemory::memcpyhh(new_data, old_data, new_size);
	}
	hmemory::free(old_data);
	return new_data;
}


int dmemory::free(Complex* data) {
	/*
	Free device memory block.
	*/
	if (data != NULL) {
		cudaFree(data);
		if (cudaGetLastError() != cudaSuccess) {
			throw_error(CUDA_FAIL_FREE_MEMORY_D);
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
		throw_error(CUDA_FAIL_ALLOCATE_MEMORY_D);
	}
	if (zero_initialise) {
		cudaMemset(data, 0, size);
		if (cudaGetLastError() != cudaSuccess) {
			throw_error(CUDA_FAIL_SET_MEMORY_D);
		}
	}
	return data;
}

Complex* dmemory::realloc(Complex* old_data, long new_size, long old_size, bool zero_initialise_if_grow) {
	/*
	Reallocate a piece of memory of size [old_size] on the device of size [new_size].
	*/
	Complex* new_data = NULL;
	if (new_size > old_size) {
		if (zero_initialise_if_grow) {
			new_data = dmemory::malloc(new_size, true);
		} else {
			new_data = dmemory::malloc(new_size, true);
		}
		dmemory::memcpydd(new_data, old_data, old_size);
	} else {
		new_data = dmemory::malloc(new_size, false);
		dmemory::memcpydd(new_data, old_data, new_size);
	}
	dmemory::free(old_data);
	return new_data;
}
