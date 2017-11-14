#include "cmemory.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <valarray>

#include "ccomplex.h"
#include "cdevice.cuh"
#include "errors.h"

using std::valarray;

template <class T>
void memory<T>::memcpydd(T* dst, T* src, long size) {
	/*
	Copy block of memory of size [size] from device location [src] to device location [dst].
	*/
	cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
	if (cudaGetLastError() != cudaSuccess) {
		throw_error(CUDA_FAIL_MEMCPY_DD);
	}
}

template <class T>
void memory<T>::memcpydh(T* dst, T* src, long size) {
	/*
	Copy block of memory of size [size] from device location [src] to host location [dst].
	*/
	cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
	if (cudaGetLastError() != cudaSuccess) {
		throw_error(CUDA_FAIL_MEMCPY_DH);
	}
}

template <class T>
void memory<T>::memcpyhd(T* dst, T* src, long size) {
	/*
	Copy block of memory of size [size] from host location [src] to device location [dst].
	*/
	cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
	if (cudaGetLastError() != cudaSuccess) {
		throw_error(CUDA_FAIL_MEMCPY_HD);
	}
}

template <class T>
void memory<T>::memcpyhh(T* dst, T* src, long size) {
	/*
	Copy block of memory of size [size] from host location [src] to host location [dst].
	*/
	cudaMemcpy(dst, src, size, cudaMemcpyHostToHost);
	if (cudaGetLastError() != cudaSuccess) {
		throw_error(CUDA_FAIL_MEMCPY_HH);
	}
}


template <class T> 
void hmemory<T>::free(T* data) {
	/*
	Free host memory block.
	*/
	if (data != NULL) {
		cudaFreeHost(data);
		if (cudaGetLastError() != cudaSuccess) {
			throw_error(CUDA_FAIL_FREE_MEMORY_H);
		}
	}
}

template <class T> 
T* hmemory<T>::malloc(long size, bool zero_initialise) {
	/*
	Allocate a piece of memory on the host of size [size].
	*/
	T* data = NULL;
	cudaMallocHost(&(data), size);
	if (cudaGetLastError() != cudaSuccess) {
		throw_error(CUDA_FAIL_ALLOCATE_MEMORY_H);
	}
	if (zero_initialise) {
		memset(data, 0, size);
	}
	return data;
}

template <class T> 
T* hmemory<T>::realloc(T* old_data, long new_size, long old_size, bool zero_initialise_if_grow) {
	/*
	Reallocate a piece of memory of size [old_size] on the host of size [new_size].
	*/
	T* new_data = NULL;
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


template <class T> 
void dmemory<T>::free(T* data) {
	/*
	Free device memory block.
	*/
	if (data != NULL) {
		cudaFree(data);
		if (cudaGetLastError() != cudaSuccess) {
			throw_error(CUDA_FAIL_FREE_MEMORY_D);
		}
	}
}

template <class T> 
T* dmemory<T>::malloc(long size, bool zero_initialise) {
	/*
	Allocate a piece of memory on the device of size [size].
	*/
	T* data = NULL;
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

template <class T> 
T* dmemory<T>::realloc(T* old_data, long new_size, long old_size, bool zero_initialise_if_grow) {
	/*
	Reallocate a piece of memory of size [old_size] on the device of size [new_size].
	*/
	T* new_data = NULL;
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

// explicit declarations required to construct classes of required types
template class memory<Complex>;
template class hmemory<Complex>;
template class dmemory<Complex>;
template class memory<Complex*>;
template class hmemory<Complex*>;
template class dmemory<Complex*>;
template class memory<Complex**>;
template class hmemory<Complex**>;
template class dmemory<Complex**>;
template class memory<int>;
template class hmemory<int>;
template class dmemory<int>;
template class memory<int*>;
template class hmemory<int*>;
template class dmemory<int*>;
