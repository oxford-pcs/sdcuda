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

int cube::clear() {
	std::memset(cube::p_data, 0, cube::memsize);
	return 0;
}

cube* cube::copy() {
	cube* datacube = new cube();
	datacube->dim = cube::dim;
	datacube->memsize = cube::memsize;
	datacube->wavelengths = cube::wavelengths;
	datacube->n_elements = cube::n_elements;
	datacube->p_data = datacube->malloc(cube::memsize, true);
	std::memcpy(datacube->p_data, cube::p_data, cube::memsize);
	for (std::vector<spslice>::iterator it = cube::slices.begin(); it != cube::slices.end(); it++) {
		long slice_idx = std::distance(cube::slices.begin(), it);
		long slice_data_idx = it->p_data - cube::p_data;
		spslice new_slice(this, &datacube->p_data[slice_data_idx], it->region, datacube->wavelengths[slice_idx]);
		datacube->slices.push_back(new_slice);
	}
	return datacube;
}

int cube::crop(std::vector<rectangle> crop_regions) {
	long x_start, y_start, x_size, y_size;
	for (int i = 0; i < cube::slices.size(); i++) {
		x_start = crop_regions[i].x_start;
		y_start = crop_regions[i].y_start;
		x_size = crop_regions[i].x_size;
		y_size = crop_regions[i].y_size;
		cube::slices[i].crop(x_start, y_start, x_size, y_size);
	}
	cube::dim[0] = NULL;
	cube::dim[1] = NULL;
	cube::n_elements = NULL;

	return 0;
}

int cube::free(Complex* data) {
	std::free(data);
	return 0;
}

Complex* cube::malloc(long size, bool zero_initialise) {
	Complex* data = NULL;
	std::malloc(sizeof(Complex)*size);
	if (zero_initialise) {
		std::memset(data, 0, sizeof(Complex)*size);
	}
	return data;
}

int cube::memcpydd(Complex* dst, Complex* src, long size) {
	cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to copy memory to device\n");
		return 1;
	}
	return 0;
}

int cube::memcpydh(Complex* dst, Complex* src, long size) {
	cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to copy memory to host\n");
		return 1;
	}
	return 0;
}

int cube::memcpyhd(Complex* dst, Complex* src, long size) {
	cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to copy memory to device\n");
		return 1;
	}
	return 0;
}

int cube::memcpyhh(Complex* dst, Complex* src, long size) {
	cudaMemcpy(dst, src, size, cudaMemcpyHostToHost);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to copy memory to host\n");
		return 1;
	}
	return 0;
}

rectangle cube::rescale(float wavelength_to_rescale_to) {
	long x_new_size, y_new_size, x_start, y_start;
	for (std::vector<spslice>::iterator it = cube::slices.begin(); it != cube::slices.end(); it++) {
		float scale_factor = wavelength_to_rescale_to / it->wavelength;
		x_new_size = round(it->region.x_size * scale_factor);
		y_new_size = round(it->region.y_size * scale_factor);
		x_start = round((it->region.x_size - x_new_size) / 2);
		y_start = round((it->region.y_size - y_new_size) / 2);
		it->crop(x_start, y_start, x_new_size, y_new_size);
	}
	cube::dim[0] = NULL;
	cube::dim[1] = NULL;
	cube::n_elements = NULL;
	return rectangle(x_start, y_start, x_new_size, y_new_size);
}


hcube::hcube(std::valarray<double> data, std::vector<long> dim, std::vector<double> wavelengths) {
	hcube::dim = dim;
	hcube::wavelengths = wavelengths;
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
			spslice new_slice(this, &hcube::p_data[i], rectangle(0, 0, hcube::dim[0], hcube::dim[1]), hcube::wavelengths[slice_idx]);
			hcube::slices.push_back(new_slice);
		}
	}
}

hcube::hcube(dcube* d_datacube) {
	hcube::dim = d_datacube->dim;
	hcube::memsize = d_datacube->memsize;
	hcube::wavelengths = d_datacube->wavelengths;
	hcube::n_elements = d_datacube->n_elements;
	hcube::p_data = hcube::malloc(hcube::memsize, true);
	hcube::memcpydh(hcube::p_data, d_datacube->p_data, hcube::memsize);
	for (std::vector<spslice>::iterator it = d_datacube->slices.begin(); it != d_datacube->slices.end(); it++) {
		long slice_idx = std::distance(d_datacube->slices.begin(), it);
		long slice_data_idx = it->p_data - d_datacube->p_data;
		spslice new_slice(this, &hcube::p_data[slice_data_idx], it->region, hcube::wavelengths[slice_idx]);
		hcube::slices.push_back(new_slice);
	}
}

hcube::~hcube() {
	hcube::free(hcube::p_data);

}

int hcube::clear() {
	memset(hcube::p_data, 0, hcube::memsize);
	return 0;
}

hcube* hcube::copy() {
	hcube* datacube = new hcube();
	datacube->dim = hcube::dim;
	datacube->memsize = hcube::memsize;
	datacube->wavelengths = hcube::wavelengths;
	datacube->n_elements = hcube::n_elements;
	datacube->p_data = datacube->malloc(hcube::memsize, true);
	datacube->memcpyhh(datacube->p_data, hcube::p_data, hcube::memsize);
	for (std::vector<spslice>::iterator it = hcube::slices.begin(); it != hcube::slices.end(); it++) {
		long slice_idx = std::distance(hcube::slices.begin(), it);
		long slice_data_idx = it->p_data - hcube::p_data;
		spslice new_slice(this, &datacube->p_data[slice_data_idx], it->region, datacube->wavelengths[slice_idx]);
		datacube->slices.push_back(new_slice);
	}
	return datacube;
}

int hcube::free(Complex* data) {
	if (data != NULL) {
		cudaFreeHost(data);
		if (cudaGetLastError() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to free memory on host\n");
		}
	}
	return 0;
}

std::valarray<double> hcube::getDataAsValarray(complex_part part) {
	std::valarray<double> data(hcube::n_elements);
	long data_offset = 0;
	for (std::vector<spslice>::iterator it = hcube::slices.begin(); it != hcube::slices.end(); it++) {
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

Complex* hcube::malloc(long size, bool zero_initialise) {
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

int hcube::write(complex_part part, string out_filename, bool clobber) {
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
	dcube::dim = h_datacube->dim;
	dcube::memsize = h_datacube->memsize;
	dcube::wavelengths = h_datacube->wavelengths;
	dcube::n_elements = h_datacube->n_elements;
	dcube::p_data = dcube::malloc(dcube::memsize, true);
	dcube::memcpyhd(dcube::p_data, h_datacube->p_data, dcube::memsize);
	for (std::vector<spslice>::iterator it = h_datacube->slices.begin(); it != h_datacube->slices.end(); it++) {
		long slice_idx = std::distance(h_datacube->slices.begin(), it);
		long slice_data_idx = it->p_data - h_datacube->p_data;
		spslice new_slice(this, &dcube::p_data[slice_data_idx], it->region, dcube::wavelengths[slice_idx]);
		dcube::slices.push_back(new_slice);
	}
}

int dcube::clear() {
	cudaMemset(dcube::p_data, 0, dcube::memsize);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to memset\n");
	}
	return 0;
}

dcube* dcube::copy() {
	dcube* datacube = new dcube();
	datacube->dim = dcube::dim;
	datacube->memsize = dcube::memsize;
	datacube->wavelengths = dcube::wavelengths;
	datacube->n_elements = dcube::n_elements;
	datacube->p_data = datacube->malloc(dcube::memsize, true);
	datacube->memcpyhh(datacube->p_data, dcube::p_data, dcube::memsize);
	for (std::vector<spslice>::iterator it = dcube::slices.begin(); it != dcube::slices.end(); it++) {
		long slice_idx = std::distance(dcube::slices.begin(), it);
		long slice_data_idx = it->p_data - dcube::p_data;
		spslice new_slice(this, &datacube->p_data[slice_data_idx], it->region, datacube->wavelengths[slice_idx]);
		datacube->slices.push_back(new_slice);
	}
	return datacube;
}

int dcube::fft(bool inverse) {
	int DIRECTION = inverse == true ? CUFFT_FORWARD : CUFFT_INVERSE;
	for (std::vector<spslice>::iterator it = dcube::slices.begin(); it != dcube::slices.end(); it++) {
		cufftHandle plan;
		if (cufftPlan2d(&plan, it->region.x_size, it->region.y_size, CUFFT_Z2Z) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT Error: Unable to create plan\n");
		}
		if (cufftExecZ2Z(plan, it->p_data, it->p_data, DIRECTION) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
		}
		if (cudaThreadSynchronize() != cudaSuccess){
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}
		cufftDestroy(plan);
	}
	return 0;
}

int dcube::free(Complex* data) {
	if (data != NULL) {
		cudaFree(data);
		if (cudaGetLastError() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to free memory on device\n");
		}
	}
	return 0;
}

Complex* dcube::malloc(long size, bool zero_initialise) {
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





