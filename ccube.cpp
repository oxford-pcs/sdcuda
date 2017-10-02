#include "ccube.h"

#include <string.h>
#include <math.h>
#include <vector>

#include <CCfits>

#include <cuda_runtime.h>
#include <cufft.h>

#include "cspslice.h"
#include "ccomplex.cuh"
#include "regions.h"

using namespace CCfits;
using std::valarray;

hcube::hcube(std::valarray<double> data, std::vector<long> dim, std::vector<double> wavelengths) {
	/*
	Construct a cube in host memory using a double valarray [data] with dimensions [dim] and 
	wavelengths [wavelengths].
	*/
	hcube::dim = dim;
	hcube::wavelengths = wavelengths;
	hcube::domain = SPATIAL;
	hcube::n_elements = std::accumulate(begin(hcube::dim), end(hcube::dim), 1, std::multiplies<long>());
	hcube::memsize = n_elements*sizeof(Complex);
	hcube::p_data = hcube::malloc(hcube::memsize, true);
	for (int i = 0; i < hcube::n_elements; i++) {
		Complex val;
		val.x = (double)data[i];
		val.y = 0.;
		hcube::p_data[i] = val;
		if (i % (hcube::dim[0] * hcube::dim[1]) == 0) {
			long slice_idx = i / (hcube::dim[0] * hcube::dim[1]);
			hspslice new_slice(this, &hcube::p_data[i], rectangle(0, 0, hcube::dim[0], hcube::dim[1]), hcube::wavelengths[slice_idx]);
			hcube::slices.push_back(new_slice);
		}
	}
}

hcube::hcube(std::valarray<Complex> data, std::vector<long> dim, std::vector<double> wavelengths, domains domain) {
	/*
	Construct a cube in host memory using a Complex valarray [data] (in domain [domain]) with dimensions [dim] and
	wavelengths [wavelengths].
	*/
	hcube::dim = dim;
	hcube::wavelengths = wavelengths;
	hcube::domain = domain;
	hcube::n_elements = std::accumulate(begin(hcube::dim), end(hcube::dim), 1, std::multiplies<long>());
	hcube::memsize = n_elements*sizeof(Complex);
	hcube::p_data = hcube::malloc(hcube::memsize, true);
	for (int i = 0; i < hcube::n_elements; i++) {
		hcube::p_data[i] = data[i];
		if (i % (hcube::dim[0] * hcube::dim[1]) == 0) {
			long slice_idx = i / (hcube::dim[0] * hcube::dim[1]);
			hspslice new_slice(this, &hcube::p_data[i], rectangle(0, 0, hcube::dim[0], hcube::dim[1]), hcube::wavelengths[slice_idx]);
			hcube::slices.push_back(new_slice);
		}
	}
}

hcube::hcube(dcube* d_datacube) {
	/*
	Construct a cube in host memory by copying a cube [d_datacube] from device memory.
	*/
	hcube::dim = d_datacube->dim;
	hcube::memsize = d_datacube->memsize;
	hcube::wavelengths = d_datacube->wavelengths;
	hcube::n_elements = d_datacube->n_elements;
	hcube::p_data = hcube::malloc(hcube::memsize, true);
	hcube::memcpydh(hcube::p_data, d_datacube->p_data, hcube::memsize);
	for (std::vector<dspslice>::iterator it = d_datacube->slices.begin(); it != d_datacube->slices.end(); it++) {
		long slice_idx = std::distance(d_datacube->slices.begin(), it);
		long slice_data_idx = it->p_data - d_datacube->p_data;
		hspslice new_slice(this, &hcube::p_data[slice_data_idx], it->region, hcube::wavelengths[slice_idx]);
		hcube::slices.push_back(new_slice);
	}
}

hcube::~hcube() {
	hcube::free(hcube::p_data);

}

int hcube::clear() {
	/*
	Clear data from host cube.
	*/
	memset(hcube::p_data, 0, hcube::memsize);
	return 0;
}

hcube* hcube::copy() {
	/*
	Deep copy an instance of a host cube.
	*/
	hcube* datacube = new hcube();
	datacube->dim = hcube::dim;
	datacube->memsize = hcube::memsize;
	datacube->wavelengths = hcube::wavelengths;
	datacube->domain = hcube::domain;
	datacube->n_elements = hcube::n_elements;
	datacube->p_data = datacube->malloc(hcube::memsize, true);
	datacube->memcpyhh(datacube->p_data, hcube::p_data, hcube::memsize);
	for (std::vector<hspslice>::iterator it = hcube::slices.begin(); it != hcube::slices.end(); it++) {
		long slice_idx = std::distance(hcube::slices.begin(), it);
		long slice_data_idx = it->p_data - hcube::p_data;
		hspslice new_slice(this, &datacube->p_data[slice_data_idx], it->region, datacube->wavelengths[slice_idx]);
		datacube->slices.push_back(new_slice);
	}
	return datacube;
}

int hcube::crop(std::vector<rectangle> crop_regions) {
	/*
	Crop each slice of a host cube by the corresponding indexed region in [crop_regions].

	This operation is in itself "unsafe", in the sense that it is possible to make the cube inconsistent
	by ending up with slices of differing sizes by using different sized regions, we therefore require 
	the user to explicitly state what the new dimensions should be.
	*/
	long x_start, y_start, x_size, y_size;
	for (int i = 0; i < hcube::slices.size(); i++) {
		x_start = crop_regions[i].x_start;
		y_start = crop_regions[i].y_start;
		x_size = crop_regions[i].x_size;
		y_size = crop_regions[i].y_size;
		hcube::slices[i].crop(x_start, y_start, x_size, y_size);
	}
	hcube::dim[0] = NULL;
	hcube::dim[1] = NULL;
	hcube::n_elements = NULL;

	return 0;
}

std::valarray<double> hcube::getDataAsValarray(complex_part part) {
	/*
	Get data from host cube corresponding to part [part].
	*/
	std::valarray<double> data(hcube::n_elements);
	long data_offset = 0;
	for (std::vector<hspslice>::iterator it = hcube::slices.begin(); it != hcube::slices.end(); it++) {
		for (int i = 0; i < it->n_elements; i++) {
			if (part == REAL) {
				data[i + data_offset] = it->p_data[i].x;
			}
			else if (part == IMAGINARY) {
				data[i + data_offset] = it->p_data[i].y;
			}
			else if (part == AMPLITUDE) {
				data[i + data_offset] = cGetAmplitude(it->p_data[i]);
			}
			else if (part == PHASE) {
				data[i + data_offset] = cGetPhase(it->p_data[i]);
			}
		}
		data_offset += it->n_elements;
	}
	return data;
}

int hcube::rescale(float wavelength_to_rescale_to, rectangle& rect) {
	/*
	Rescale slices of a device cube to the wavelength [wavelength_to_rescale_to]. A rectangle representing the 
	slice with the smallest region [smallest_region] is populated.
	
	Note that this only makes sense when working on data in the frequency domain and so a CCUBE_FFT_BAD_DOMAIN 
	error will be thrown if [domain] is SPATIAL.
	*/
	if (hcube::domain == FREQUENCY) {
		long x_new_size, y_new_size, x_start, y_start;
		for (std::vector<hspslice>::iterator it = hcube::slices.begin(); it != hcube::slices.end(); it++) {
			float scale_factor = wavelength_to_rescale_to / it->wavelength;
			x_new_size = round(it->region.x_size * scale_factor);
			y_new_size = round(it->region.y_size * scale_factor);
			x_start = round((it->region.x_size - x_new_size) / 2);
			y_start = round((it->region.y_size - y_new_size) / 2);
			it->crop(x_start, y_start, x_new_size, y_new_size);
		}
		hcube::dim[0] = NULL;
		hcube::dim[1] = NULL;
		hcube::n_elements = NULL;
		rect = rectangle(x_start, y_start, x_new_size, y_new_size);
	} else {
		throw_error(CCUBE_FFT_BAD_DOMAIN);
	}
	return 0;
}

int hcube::write(complex_part part, string out_filename, bool clobber) {
	/*
	Write hard copy of complex part [part] of host datacube to file [out_filename].
	*/
	long naxis = hcube::dim.size();
	long naxes[3] = { hcube::dim[0], hcube::dim[1], hcube::dim[2] };
	long n_elements = std::accumulate(begin(hcube::dim), end(hcube::dim), 1, std::multiplies<long>());
	std::auto_ptr<FITS> pFits(0);
	try {
		std::string fileName(out_filename);
		if (clobber) {
			fileName.insert(0, std::string("!"));
		}
		pFits.reset(new FITS(fileName, DOUBLE_IMG, naxis, naxes));
	}
	catch (FITS::CantCreate) {
		return -1;
	}

	pFits->addImage("DATA", DOUBLE_IMG, std::vector<long> ({ hcube::dim[0], hcube::dim[1] }));

	long fpixel(1);
	pFits->pHDU().write(fpixel, n_elements, hcube::getDataAsValarray(part));

	return 0;
}


dcube::dcube(hcube* h_datacube) {
	/*
	Construct a cube in device memory by copying a cube [h_datacube] from host memory.
	*/
	dcube::dim = h_datacube->dim;
	dcube::memsize = h_datacube->memsize;
	dcube::wavelengths = h_datacube->wavelengths;
	dcube::domain = h_datacube->domain;
	dcube::n_elements = h_datacube->n_elements;
	dcube::p_data = dcube::malloc(dcube::memsize, true);
	dcube::memcpyhd(dcube::p_data, h_datacube->p_data, dcube::memsize);
	for (std::vector<hspslice>::iterator it = h_datacube->slices.begin(); it != h_datacube->slices.end(); it++) {
		long slice_idx = std::distance(h_datacube->slices.begin(), it);
		long slice_data_idx = it->p_data - h_datacube->p_data;
		dspslice new_slice(this, &dcube::p_data[slice_data_idx], it->region, dcube::wavelengths[slice_idx]);
		dcube::slices.push_back(new_slice);
	}
}

int dcube::clear() {
	/*
	Clear data from device cube.
	*/
	cudaMemset(dcube::p_data, 0, dcube::memsize);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to memset\n");
	}
	return 0;
}

dcube* dcube::copy() {
	/*
	Deep copy an instance of a device cube.
	*/
	dcube* datacube = new dcube();
	datacube->dim = dcube::dim;
	datacube->memsize = dcube::memsize;
	datacube->wavelengths = dcube::wavelengths;
	datacube->domain = dcube::domain;
	datacube->n_elements = dcube::n_elements;
	datacube->p_data = datacube->malloc(dcube::memsize, true);
	datacube->memcpyhh(datacube->p_data, dcube::p_data, dcube::memsize);
	for (std::vector<dspslice>::iterator it = dcube::slices.begin(); it != dcube::slices.end(); it++) {
		long slice_idx = std::distance(dcube::slices.begin(), it);
		long slice_data_idx = it->p_data - dcube::p_data;
		dspslice new_slice(this, &datacube->p_data[slice_data_idx], it->region, datacube->wavelengths[slice_idx]);
		datacube->slices.push_back(new_slice);
	}
	return datacube;
}

int dcube::crop(std::vector<rectangle> crop_regions) {
	/*
	Crop each slice of a device cube by the corresponding indexed region in [crop_regions].
	*/
	long x_start, y_start, x_size, y_size;
	for (int i = 0; i < dcube::slices.size(); i++) {
		x_start = crop_regions[i].x_start;
		y_start = crop_regions[i].y_start;
		x_size = crop_regions[i].x_size;
		y_size = crop_regions[i].y_size;
		dcube::slices[i].crop(x_start, y_start, x_size, y_size);
	}
	dcube::dim[0] = NULL;
	dcube::dim[1] = NULL;
	dcube::n_elements = NULL;

	return 0;
}

int dcube::fft(bool inverse) {
	/* 
	Perform a fast fourier transform on the device data in the direction specified by the [inverse] flag.
	*/
	int DIRECTION = inverse == true ? CUFFT_FORWARD : CUFFT_INVERSE;
	for (std::vector<dspslice>::iterator it = dcube::slices.begin(); it != dcube::slices.end(); it++) {
		cufftHandle plan;
		if (cufftPlan2d(&plan, it->region.x_size, it->region.y_size, CUFFT_Z2Z) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT Error: Unable to create plan\n");
		}
		if (cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(it->p_data), 
			reinterpret_cast<cufftDoubleComplex*>(it->p_data), DIRECTION) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
		}
		if (cudaThreadSynchronize() != cudaSuccess){
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}
		cufftDestroy(plan);
	}
	switch (dcube::domain) {
	case (SPATIAL) :
		dcube::domain = FREQUENCY;
		break;
	case (FREQUENCY) :
		dcube::domain = SPATIAL;
		break;
	}
	return 0;
}

int dcube::rescale(float wavelength_to_rescale_to, rectangle &smallest_region) {
	/*
	Rescale slices of a device cube to the wavelength [wavelength_to_rescale_to]. A rectangle representing the 
	slice with the smallest region [smallest_region] is populated.
	
	Note that this only makes sense when working on data in the frequency domain and so a CCUBE_FFT_BAD_DOMAIN 
	error will be thrown if [domain] is SPATIAL.
	*/
	if (dcube::domain == FREQUENCY) {
		long x_new_size, y_new_size, x_start, y_start;
		for (std::vector<dspslice>::iterator it = dcube::slices.begin(); it != dcube::slices.end(); it++) {
			float scale_factor = wavelength_to_rescale_to / it->wavelength;
			x_new_size = round(it->region.x_size * scale_factor);
			y_new_size = round(it->region.y_size * scale_factor);
			x_start = round((it->region.x_size - x_new_size) / 2);
			y_start = round((it->region.y_size - y_new_size) / 2);
			it->crop(x_start, y_start, x_new_size, y_new_size);
		}
		dcube::dim[0] = NULL;
		dcube::dim[1] = NULL;
		dcube::n_elements = NULL;
		smallest_region = rectangle(x_start, y_start, x_new_size, y_new_size);
	}
	else {
		throw_error(CCUBE_FFT_BAD_DOMAIN);
	}
	return 0;
	
}
