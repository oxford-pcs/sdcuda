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


hspslice::hspslice(hcube* datacube, std::valarray<double> data, rectangle region, double wavelength) {
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

hspslice::hspslice(hcube* datacube, std::valarray<Complex> data, rectangle region, double wavelength) {
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

int hspslice::clear() {
	/*
	Clear data from host slice.
	*/
	memset(hspslice::p_data, 0, hspslice::memsize);
	return 0;
}

int hspslice::crop(rectangle new_region) {
	/*
	Crop a slice in host memory to the region [region], making the data contiguous within the slice.
	*/
	rectangle* old_region = &(hspslice::region);
	for (int row = 0; row < new_region.y_size; row++) {
		hspslice::memcpydd(&p_data[(row*new_region.x_size)], &p_data[((row + (new_region.y_start - old_region->y_start))*old_region->x_size) +
			(new_region.x_start - old_region->x_start)], new_region.x_size*sizeof(Complex));
	}
	hspslice::region = new_region;
	long new_memsize = new_region.x_size*new_region.y_size*sizeof(Complex);
	hspslice::p_data = hspslice::realloc(hspslice::p_data, new_memsize, hspslice::memsize, false);
	hspslice::memsize = new_memsize;
	return 0;
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

int hspslice::grow(rectangle region) {
	/*
	Grow a slice in host memory to the region [region], making the data contiguous within the slice.
	*/
	long x_start = region.x_start;
	long y_start = region.y_start;
	long x_size = region.x_size;
	long y_size = region.y_size;
	long new_memsize = x_size*y_size*sizeof(Complex);
	hspslice::p_data = hspslice::realloc(hspslice::p_data, new_memsize, hspslice::memsize, false);
	for (int row = y_size - 1; row == 0; row--) {
		hspslice::memcpyhh(&p_data[((row + y_start)*hspslice::region.x_size) + x_start], &p_data[(row*x_size)], x_size*sizeof(Complex));
	}
	hspslice::region = rectangle(x_start, y_start, x_size, y_size);
	hspslice::memsize = new_memsize;
	return 0;
}


dspslice::dspslice(dcube* datacube, std::valarray<double> data, rectangle region, double wavelength) {
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

dspslice::dspslice(dcube* datacube, std::valarray<Complex> data, rectangle region, double wavelength) {
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

int dspslice::clear() {
	/*
	Clear data from device slice.
	*/
	cudaMemset(dspslice::p_data, 0, dspslice::memsize);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to memset\n");
	}
	return 0;
}

int dspslice::crop(rectangle new_region) {
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
	return 0;
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

int dspslice::grow(rectangle new_region) {
	/*
	Grow a slice in device memory to the region [region], making the data contiguous within the slice.
	*/
	rectangle* old_region = &(dspslice::region);
	long new_memsize = new_region.x_size*new_region.y_size*sizeof(Complex);
	dspslice::p_data = dspslice::realloc(dspslice::p_data, new_memsize, dspslice::memsize, true);
	for (int row = old_region->y_size - 1; row >= 0; row--) {
		dspslice::memcpydd(&dspslice::p_data[((row + (old_region->y_start - new_region.y_start))*new_region.x_size) +
			(old_region->x_start - new_region.x_start)], &dspslice::p_data[(row*old_region->x_size)], old_region->x_size*sizeof(Complex));
	}
	dspslice::region = new_region;
	dspslice::memsize = new_memsize;
	return 0;
}
