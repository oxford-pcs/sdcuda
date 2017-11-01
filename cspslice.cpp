#include "cspslice.h"

#include <stdio.h>
#include <vector>
#include <numeric>

#include <cuda_runtime.h>
#include <cufft.h>

#include "ccube.h"
#include "regions.h"

long2 spslice::getDimensions() {
	return long2 { spslice::region.x_size, spslice::region.y_size };
}

long spslice::getNumberOfElements() {
	return spslice::region.x_size * spslice::region.y_size;
}


hspslice::hspslice(hcube* datacube, std::valarray<double> data, rectangle region, int wavelength) {
	/*
	Construct a slice in host memory from a datacube [datacube] with the data pointer [p_data] of
	region [region] and at wavelength [wavelength].
	*/
	hspslice::datacube = datacube;
	hspslice::region = region;
	hspslice::wavelength = wavelength;
	hspslice::memsize = hspslice::getNumberOfElements()*sizeof(Complex);
	hspslice::p_data = hspslice::malloc(hspslice::memsize, true);
	for (int i = 0; i < hspslice::getNumberOfElements(); i++) {
		hspslice::p_data[i] = Complex((double)data[i], 0);
	}
}

hspslice::hspslice(hcube* datacube, std::valarray<Complex> data, rectangle region, int wavelength) {
	/*
	Construct a slice in host memory from a datacube [datacube] with the data pointer [p_data] of
	region [region] and at wavelength [wavelength].
	*/
	hspslice::datacube = datacube;
	hspslice::region = region;
	hspslice::wavelength = wavelength;
	hspslice::memsize = hspslice::getNumberOfElements()*sizeof(Complex);
	hspslice::p_data = hspslice::malloc(hspslice::memsize, true);
	for (int i = 0; i < hspslice::getNumberOfElements(); i++) {
		hspslice::p_data[i] = data[i];
	}
}

hspslice::hspslice(hcube* datacube, dspslice* dspslice) {
	/*
	Construct a slice in host memory using a slice in device memory.
	*/
	hspslice::datacube = datacube;
	hspslice::region = dspslice->region;
	hspslice::memsize = dspslice->memsize;
	hspslice::wavelength = dspslice->wavelength;
	hspslice::p_data = hspslice::malloc(hspslice::memsize, true);
	hspslice::memcpydh(hspslice::p_data, dspslice->p_data, dspslice->memsize);
}

hspslice::~hspslice() {
	hspslice::free(p_data);
}

void hspslice::clear() {
	/*
	Clear data from host slice.
	*/
	memset(hspslice::p_data, 0, hspslice::memsize);
}

void hspslice::crop(rectangle new_region) {
	/*
	Crop a slice in host memory to the region [region], making the data contiguous within the slice.
	*/
	rectangle* old_region = &(hspslice::region);
	for (int row = 0; row < new_region.y_size; row++) {
		hspslice::memcpyhh(&hspslice::p_data[(row*new_region.x_size)], &hspslice::p_data[((row + (new_region.y_start - old_region->y_start))*old_region->x_size) +
			(new_region.x_start - old_region->x_start)], new_region.x_size*sizeof(Complex));
	}
	hspslice::region = new_region;
	long new_memsize = new_region.x_size*new_region.y_size*sizeof(Complex);
	hspslice::p_data = hspslice::realloc(hspslice::p_data, new_memsize, hspslice::memsize, false);
	hspslice::memsize = new_memsize;
}

hspslice* hspslice::deepcopy() {
	/*
	Deep copy an instance of a host slice.
	*/
	hspslice* new_slice = new hspslice();
	new_slice->datacube = hspslice::datacube;
	new_slice->region = hspslice::region;
	new_slice->memsize = hspslice::memsize;
	new_slice->wavelength = hspslice::wavelength;
	new_slice->p_data = hspslice::malloc(hspslice::memsize, true);
	new_slice->memcpyhh(new_slice->p_data, hspslice::p_data, hspslice::memsize);
	return new_slice;
}

void hspslice::grow(rectangle new_region) {
	/*
	Grow a slice in host memory to the region [region], making the data contiguous within the slice. A temporary
	row is required as the data that is copied into the realloc'ed array needs to be erased after it's been moved to
	the correct position in the array, whilst avoiding zeroing data that's been moved.
	*/
	rectangle* old_region = &(hspslice::region);
	long new_row_memsize = new_region.x_size*sizeof(Complex);
	long new_memsize = new_row_memsize*new_region.y_size;
	hspslice::p_data = hspslice::realloc(hspslice::p_data, new_memsize, hspslice::memsize, true);
	Complex* row_data_tmp = hspslice::malloc(old_region->x_size*sizeof(Complex), true);
	for (int row = old_region->y_size - 1; row >= 0; row--) {
		dspslice::memcpyhh(row_data_tmp, &hspslice::p_data[(row*old_region->x_size)], old_region->x_size*sizeof(Complex));
		std::memset(&hspslice::p_data[(row*old_region->x_size)], 0, new_row_memsize);
		dspslice::memcpyhh(&hspslice::p_data[((row + (old_region->y_start - new_region.y_start))*new_region.x_size) +
			(old_region->x_start - new_region.x_start)], row_data_tmp, old_region->x_size*sizeof(Complex));
	}
	hspslice::free(row_data_tmp);
	hspslice::region = new_region;
	hspslice::memsize = new_memsize;
}


dspslice::dspslice(dcube* datacube, std::valarray<double> data, rectangle region, int wavelength) {
	/*
	Construct a slice in device memory from a datacube [datacube] with the data pointer [p_data] of
	region [region] and at wavelength [wavelength].
	*/
	dspslice::datacube = datacube;
	dspslice::region = region;
	dspslice::wavelength = wavelength;
	dspslice::memsize = dspslice::getNumberOfElements()*sizeof(Complex);
	dspslice::p_data = dspslice::malloc(dspslice::memsize, true);
	for (int i = 0; i < dspslice::getNumberOfElements(); i++) {
		dspslice::p_data[i] = Complex((double)data[i], 0);
	}
}

dspslice::dspslice(dcube* datacube, std::valarray<Complex> data, rectangle region, int wavelength) {
	/*
	Construct a slice in device memory from a datacube [datacube] with the data pointer [p_data] of
	region [region] and at wavelength [wavelength].
	*/
	dspslice::datacube = datacube;
	dspslice::region = region;
	dspslice::wavelength = wavelength;
	dspslice::memsize = dspslice::getNumberOfElements()*sizeof(Complex);
	dspslice::p_data = dspslice::malloc(dspslice::memsize, true);
	for (int i = 0; i < dspslice::getNumberOfElements(); i++) {
		dspslice::p_data[i] = data[i];
	}
}

dspslice::dspslice(dcube* datacube, hspslice* hspslice) {
	/*
	Construct a slice in device memory using a slice in host memory.
	*/
	dspslice::datacube = datacube;
	dspslice::region = hspslice->region;
	dspslice::memsize = hspslice->memsize;
	dspslice::wavelength = hspslice->wavelength;
	dspslice::p_data = dspslice::malloc(dspslice::memsize, true);
	dspslice::memcpyhd(dspslice::p_data, hspslice->p_data, hspslice->memsize);
}

dspslice::~dspslice() {
	dspslice::free(p_data);
}

void dspslice::clear() {
	/*
	Clear data from device slice.
	*/
	cudaMemset(dspslice::p_data, 0, dspslice::memsize);
	if (cudaGetLastError() != cudaSuccess) {
		throw_error(CUDA_FAIL_SET_MEMORY_D);
	}
}

void dspslice::crop(rectangle new_region) {
	/*
	Crop a slice in device memory to the region [region], making the data contiguous within the slice.
	*/
	rectangle* old_region = &(dspslice::region);
	for (int row = 0; row < new_region.y_size; row++) {
		dspslice::memcpydd(&dspslice::p_data[(row*new_region.x_size)], &dspslice::p_data[((row + (new_region.y_start - old_region->y_start))*old_region->x_size) +
			(new_region.x_start - old_region->x_start)], new_region.x_size*sizeof(Complex));
	}
	dspslice::region = new_region;
	long new_memsize = new_region.x_size*new_region.y_size*sizeof(Complex);
	dspslice::p_data = dspslice::realloc(dspslice::p_data, new_memsize, dspslice::memsize, false);
	dspslice::memsize = new_memsize;
}

dspslice* dspslice::deepcopy() {
	/*
	Deep copy an instance of a device slice.
	*/
	dspslice* new_slice = new dspslice();
	new_slice->datacube = dspslice::datacube;
	new_slice->region = dspslice::region;
	new_slice->memsize = dspslice::memsize;
	new_slice->wavelength = dspslice::wavelength;
	new_slice->p_data = dspslice::malloc(dspslice::memsize, true);
	new_slice->memcpydd(new_slice->p_data, dspslice::p_data, dspslice::memsize);
	return new_slice;
}

void dspslice::grow(rectangle new_region) {
	/*
	Grow a slice in device memory to the region [region], making the data contiguous within the slice. A temporary
	row is required as the data that is copied into the realloc'ed array needs to be erased after it's been moved to 
	the correct position in the array, whilst avoiding zeroing data that's been moved.
	*/
	rectangle* old_region = &(dspslice::region);
	long new_row_memsize = new_region.x_size*sizeof(Complex);
	long new_memsize = new_row_memsize*new_region.y_size;
	dspslice::p_data = dspslice::realloc(dspslice::p_data, new_memsize, dspslice::memsize, true);
	Complex* row_data_tmp = dspslice::malloc(old_region->x_size*sizeof(Complex), true);
	for (int row = old_region->y_size - 1; row >= 0; row--) {
		dspslice::memcpydd(row_data_tmp, &dspslice::p_data[(row*old_region->x_size)], old_region->x_size*sizeof(Complex));
		cudaMemset(&dspslice::p_data[(row*old_region->x_size)], 0, new_row_memsize);
		dspslice::memcpydd(&dspslice::p_data[((row + (old_region->y_start - new_region.y_start))*new_region.x_size) +
			(old_region->x_start - new_region.x_start)], row_data_tmp, old_region->x_size*sizeof(Complex));
	}
	dspslice::free(row_data_tmp);
	dspslice::region = new_region;
	dspslice::memsize = new_memsize;
}
