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
#include "errors.h"

using namespace CCfits;
using std::valarray;

hcube::hcube(std::valarray<double> data, std::vector<long> dim, std::vector<int> wavelengths) {
	/*
	Construct a cube in host memory using a double valarray [data] with dimensions [dim] and 
	wavelengths [wavelengths].
	*/
	hcube::domain = SPATIAL;
	long slice_nelements = dim[0] * dim[1];
	long offset;
	for (int i = 0; i < dim[2]; i++) {
		offset = i*slice_nelements;
		hspslice* new_slice = new hspslice(this, data[std::slice(offset, slice_nelements, 1)], rectangle(0, 0, dim[0], dim[1]), wavelengths[i]);
		hcube::slices.push_back(new_slice);
	}
}

hcube::hcube(std::valarray<Complex> data, std::vector<long> dim, std::vector<int> wavelengths, ccube_domains domain) {
	/*
	Construct a cube in host memory using a double valarray [data] of domain [domain] with dimensions [dim] and
	wavelengths [wavelengths].
	*/
	hcube::domain = domain;
	long slice_n_elements = dim[0] * dim[1];
	long offset;
	for (int i = 0; i < dim[2]; i++) {
		offset = i*slice_n_elements;
		hspslice* new_slice = new hspslice(this, data[std::slice(offset, slice_n_elements, 1)], rectangle(0, 0, dim[0], dim[1]), wavelengths[i]);
		hcube::slices.push_back(new_slice);
	}
}

hcube::hcube(dcube* d_datacube) {
	/*
	Construct a cube in host memory by copying a cube [d_datacube] from device memory.
	*/
	hcube::domain = d_datacube->domain;
	hcube::state = d_datacube->state;
	for (std::vector<dspslice*>::iterator it = d_datacube->slices.begin(); it != d_datacube->slices.end(); ++it) {
		hspslice* new_slice = new hspslice(this, d_datacube->slices[std::distance(d_datacube->slices.begin(), it)]);
		hcube::slices.push_back(new_slice);
	}
}

hcube::~hcube() {
	// delete slice instances
	for (std::vector<hspslice*>::iterator it = hcube::slices.begin(); it != hcube::slices.end(); ++it) {
		delete (*it);
	}
}

int hcube::clear() {
	/*
	Clear data from all slices in host cube.
	*/
	for (std::vector<hspslice*>::iterator it = hcube::slices.begin(); it != hcube::slices.end(); ++it) {
		(*it)->clear();
	}
	return 0;
}

int hcube::crop(rectangle region) {
	/*
	Crop all slices of a host cube by the region [region].
	*/
	for (std::vector<hspslice*>::iterator it = hcube::slices.begin(); it != hcube::slices.end(); ++it) {
		(*it)->crop(region);
	}
	hcube::state = OK;
	return 0;
}

int hcube::crop(std::vector<rectangle> regions) {
	/*
	Crop each slice of a host cube by the corresponding region in vector [regions].
	*/
	std::vector<long> region_size_x, region_size_y;
	for (std::vector<hspslice*>::iterator it = hcube::slices.begin(); it != hcube::slices.end(); ++it) {
		(*it)->crop(regions[std::distance(hcube::slices.begin(), it)]);
		region_size_x.push_back(regions[std::distance(hcube::slices.begin(), it)].x_size);
		region_size_y.push_back(regions[std::distance(hcube::slices.begin(), it)].y_size);
	}

	// check cube integrity
	if (std::equal(region_size_x.begin() + 1, region_size_x.end(), region_size_x.begin()) &&
		std::equal(region_size_y.begin() + 1, region_size_y.end(), region_size_y.begin())) {
		hcube::state = OK;
	} else {
		hcube::state = INCONSISTENT;
	}
	return 0;
}

hcube* hcube::deepcopy() {
	/*
	Deep copy an instance of a host cube.
	*/
	hcube* new_datacube = new hcube();
	new_datacube->domain = hcube::domain;
	new_datacube->state = hcube::state;
	for (std::vector<hspslice*>::iterator it = hcube::slices.begin(); it != hcube::slices.end(); ++it) {
		hspslice* new_slice = (*it)->deepcopy();
		new_datacube->slices.push_back(new_slice);
	}
	return new_datacube;
}

std::valarray<double> hcube::getDataAsValarray(complex_part part) {
	/*
	Get data from host cube corresponding to complex part [part] as a valarray.
	*/
	long nelements = 0;
	for (std::vector<hspslice*>::iterator it = hcube::slices.begin(); it != hcube::slices.end(); ++it) {
		nelements += (*it)->getNumberOfElements();
	}

	std::valarray<double> data(nelements);
	long data_offset = 0;
	for (std::vector<hspslice*>::iterator it = hcube::slices.begin(); it != hcube::slices.end(); ++it) {
		for (int i = 0; i < (*it)->getNumberOfElements(); i++) {
			if (part == REAL) {
				data[i + data_offset] = (*it)->p_data[i].x;
			} else if (part == IMAGINARY) {
				data[i + data_offset] = (*it)->p_data[i].y;
			} else if (part == AMPLITUDE) {
				data[i + data_offset] = cGetAmplitude((*it)->p_data[i]);
			} else if (part == PHASE) {
				data[i + data_offset] = cGetPhase((*it)->p_data[i]);
			}
		}
		data_offset += (*it)->getNumberOfElements();
	}
	return data;
}

std::valarray<double> hcube::getDataAsValarray(complex_part part, int slice_index) {
	/*
	Get data from slice [slice_index] of host cube corresponding to complex part [part] as a valarray.
	*/
	long nelements = hcube::slices[slice_index]->region.x_size*hcube::slices[slice_index]->region.y_size;
	std::valarray<double> data(nelements);
	for (int i = 0; i < nelements; i++) {
		if (part == REAL) {
			data[i] = hcube::slices[slice_index]->p_data[i].x;
		} else if (part == IMAGINARY) {
			data[i] = hcube::slices[slice_index]->p_data[i].y;
		} else if (part == AMPLITUDE) {
			data[i] = cGetAmplitude(hcube::slices[slice_index]->p_data[i]);
		} else if (part == PHASE) {
			data[i] = cGetPhase(hcube::slices[slice_index]->p_data[i]);
		}
	}
	return data;
}

rectangle hcube::getSmallestSliceRegion() {
	std::vector<rectangle> regions;
	std::vector<long> region_sizes;
	for (std::vector<hspslice*>::iterator it = hcube::slices.begin(); it != hcube::slices.end(); ++it) {
		regions.push_back((*it)->region);
		region_sizes.push_back((*it)->region.x_size*(*it)->region.y_size);
	}
	std::vector<long>::iterator result = std::min_element(std::begin(region_sizes), std::end(region_sizes));
	return regions[std::distance(std::begin(region_sizes), result)];
}

int hcube::rescale(float wavelength_to_rescale_to) {
	/*
	Rescale slices of a host cube to the wavelength [wavelength_to_rescale_to]. Note that this only makes sense when 
	working on data in the frequency domain.
	*/
	if (hcube::domain == FREQUENCY) {
		std::vector<long> region_size_x, region_size_y;
		long x_new_size, y_new_size, x_start, y_start;
		for (std::vector<hspslice*>::iterator it = hcube::slices.begin(); it != hcube::slices.end(); ++it) {
			double scale_factor = wavelength_to_rescale_to / (double)(*it)->wavelength;
			x_new_size = round((*it)->region.x_size * scale_factor);
			y_new_size = round((*it)->region.y_size * scale_factor);
			x_start = (*it)->region.x_start + round(((*it)->region.x_size - x_new_size) / 2);
			y_start = (*it)->region.y_start + round(((*it)->region.y_size - y_new_size) / 2);
			rectangle this_region = rectangle(x_start, y_start, x_new_size, y_new_size);
			(*it)->crop(this_region);
			region_size_x.push_back(this_region.x_size);
			region_size_y.push_back(this_region.y_size);
		}
		// check cube integrity (unless all wavelengths are equal, this should fail!)
		if (std::equal(region_size_x.begin() + 1, region_size_x.end(), region_size_x.begin()) &&
			std::equal(region_size_y.begin() + 1, region_size_y.end(), region_size_y.begin())) {
			hcube::state = OK;
		} else {
			hcube::state = INCONSISTENT;
		}
	} else {
		throw_error(CCUBE_BAD_DOMAIN);
	}
	return 0;
}

int hcube::write(complex_part part, string out_filename, bool clobber) {
	/*
	Write hard copy of complex part [part] of host datacube to file [out_filename].
	*/
	if (hcube::state == OK) {
		long naxis = 3;
		long naxes[3] = { (*hcube::slices[0]).getDimensions().x, (*hcube::slices[0]).getDimensions().y, long(hcube::slices.size()) };
		long n_elements = naxes[0] * naxes[1] * naxes[2];

		std::auto_ptr<FITS> pFits(0);
		try {
			std::string fileName(out_filename);
			if (clobber) {
				fileName.insert(0, std::string("!"));
			}
			pFits.reset(new FITS(fileName, DOUBLE_IMG, naxis, naxes));
		} catch (FITS::CantCreate) {
			throw_error(CCUBE_FAIL_WRITE);
		}

		pFits->addImage("DATA", DOUBLE_IMG, std::vector<long>({ naxes[0], naxes[1] }));

		long fpixel(1);
		pFits->pHDU().write(fpixel, n_elements, hcube::getDataAsValarray(part));
	} else if (hcube::state == INCONSISTENT) {
		throw_error(CCUBE_FAIL_INTEGRITY_CHECK);
	}
	return 0;
}

int hcube::write(complex_part part, string out_filename, int slice_index, bool clobber) {
	/*
	Write hard copy of complex part [part] of slice [slice_index] from host datacube to file [out_filename].
	*/
	if (hcube::state == OK) {
		long naxis = 2;
		long naxes[2] = { (*hcube::slices[slice_index]).getDimensions().x, (*hcube::slices[slice_index]).getDimensions().y };
		long n_elements = naxes[0] * naxes[1];
		std::auto_ptr<FITS> pFits(0);
		try {
			std::string fileName(out_filename);
			if (clobber) {
				fileName.insert(0, std::string("!"));
			}
			pFits.reset(new FITS(fileName, DOUBLE_IMG, naxis, naxes));
		} catch (FITS::CantCreate) {
			throw_error(CCUBE_FAIL_WRITE);
		}

		pFits->addImage("DATA", DOUBLE_IMG, std::vector<long>({ naxes[0], naxes[1] }));

		long fpixel(1);
		pFits->pHDU().write(fpixel, n_elements, hcube::getDataAsValarray(part, slice_index));
	}
	return 0;
}


dcube::dcube(hcube* h_datacube) {
	/*
	Construct a cube in device memory by copying a cube [h_datacube] from host memory.
	*/
	dcube::domain = h_datacube->domain;
	dcube::state = h_datacube->state;
	for (std::vector<hspslice*>::iterator it = h_datacube->slices.begin(); it != h_datacube->slices.end(); ++it) {
		dspslice* new_slice = new dspslice(this, h_datacube->slices[std::distance(h_datacube->slices.begin(), it)]);
		dcube::slices.push_back(new_slice);
	}
}

dcube::~dcube() {
	// delete slice instances
	for (std::vector<dspslice*>::iterator it = dcube::slices.begin(); it != dcube::slices.end(); ++it) {
		delete (*it);
	}
}

int dcube::clear() {
	/*
	Clear data from all slices in device cube.
	*/
	for (std::vector<dspslice*>::iterator it = dcube::slices.begin(); it != dcube::slices.end(); ++it) {
		(*it)->clear();
	}
	return 0;
}

int dcube::crop(rectangle region) {
	/*
	Crop all slices of a device cube by the region [region].
	*/
	for (std::vector<dspslice*>::iterator it = dcube::slices.begin(); it != dcube::slices.end(); ++it) {
		(*it)->crop(region);
	}
	dcube::state = OK;
	return 0;
}

int dcube::crop(std::vector<rectangle> regions) {
	/*
	Crop each slice of a device cube by the corresponding region in vector [regions].
	*/
	std::vector<long> region_size_x, region_size_y;
	for (std::vector<dspslice*>::iterator it = dcube::slices.begin(); it != dcube::slices.end(); ++it) {
		(*it)->crop(regions[std::distance(dcube::slices.begin(), it)]);
		region_size_x.push_back(regions[std::distance(dcube::slices.begin(), it)].x_size);
		region_size_y.push_back(regions[std::distance(dcube::slices.begin(), it)].y_size);
	}

	// check cube integrity
	if (std::equal(region_size_x.begin() + 1, region_size_x.end(), region_size_x.begin()) &&
		std::equal(region_size_y.begin() + 1, region_size_y.end(), region_size_y.begin())) {
		dcube::state = OK;
	} else {
		dcube::state = INCONSISTENT;
	}
	return 0;
}

dcube* dcube::deepcopy() {
	/*
	Deep copy an instance of a device cube.
	*/
	dcube* new_datacube = new dcube();
	new_datacube->domain = dcube::domain;
	new_datacube->state = dcube::state;
	for (std::vector<dspslice*>::iterator it = dcube::slices.begin(); it != dcube::slices.end(); ++it) {
		dspslice* new_slice = (*it)->deepcopy();
		new_datacube->slices.push_back(new_slice);
	}
	return new_datacube;
}

int dcube::fft(bool inverse) {
	/*
	Perform a fast fourier transform on the device data in the direction specified by the [inverse] flag.
	*/
	int DIRECTION = (inverse == true) ? CUFFT_FORWARD : CUFFT_INVERSE;
	for (std::vector<dspslice*>::iterator it = dcube::slices.begin(); it != dcube::slices.end(); ++it) {
		cufftHandle plan;
		if (cufftPlan2d(&plan, (*it)->region.x_size, (*it)->region.y_size, CUFFT_Z2Z) != CUFFT_SUCCESS) {
			throw_error(CUDA_FFT_FAIL_CREATE_PLAN);
		}
		if (cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>((*it)->p_data),
			reinterpret_cast<cufftDoubleComplex*>((*it)->p_data), DIRECTION) != CUFFT_SUCCESS) {
			throw_error(CUDA_FFT_FAIL_EXECUTE_PLAN);
		}
		if (cudaThreadSynchronize() != cudaSuccess){
			throw_error(CUDA_FAIL_SYNCHRONIZE);
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

rectangle dcube::getSmallestSliceRegion() {
	std::vector<rectangle> regions;
	std::vector<long> region_sizes;
	for (std::vector<dspslice*>::iterator it = dcube::slices.begin(); it != dcube::slices.end(); ++it) {
		regions.push_back((*it)->region);
		region_sizes.push_back((*it)->region.x_size*(*it)->region.y_size);
	}
	std::vector<long>::iterator result = std::min_element(std::begin(region_sizes), std::end(region_sizes));
	return regions[std::distance(std::begin(region_sizes), result)];
}

int dcube::rescale(std::vector<double> scale_factors) {
	/*
	Rescale slices of a device cube by [factors]. Note that this only makes sense when working on data in the frequency domain.
	*/
	if (dcube::domain == FREQUENCY) {
		std::vector<long> region_size_x, region_size_y;
		long x_new_size, y_new_size, x_start, y_start;
		for (int i = 0; i < dcube::slices.size(); i++) {
			x_new_size = round(dcube::slices[i]->region.x_size * scale_factors[i]);
			y_new_size = round(dcube::slices[i]->region.y_size * scale_factors[i]);
			x_start = dcube::slices[i]->region.x_start + round((dcube::slices[i]->region.x_size - x_new_size) / 2);
			y_start = dcube::slices[i]->region.y_start + round((dcube::slices[i]->region.y_size - y_new_size) / 2);
			rectangle this_region = rectangle(x_start, y_start, x_new_size, y_new_size);
			if (this_region.x_size < dcube::slices[i]->region.x_size) {
				dcube::slices[i]->crop(this_region);
			} else if (this_region.x_size > dcube::slices[i]->region.x_size)  {
				dcube::slices[i]->grow(this_region);
			}
			region_size_x.push_back(this_region.x_size);
			region_size_y.push_back(this_region.y_size);
		}

		// check cube integrity (unless all wavelengths are equal, this should fail!)
		if (std::equal(region_size_x.begin() + 1, region_size_x.end(), region_size_x.begin()) &&
			std::equal(region_size_y.begin() + 1, region_size_y.end(), region_size_y.begin())) {
			dcube::state = OK;
		} else {
			dcube::state = INCONSISTENT;
		}
	} else {
		throw_error(CCUBE_BAD_DOMAIN);
	}
	return 0;
}


