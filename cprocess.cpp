#include "cprocess.h"

#include "ccube.h"
#include "cinput.h"
#include "cclparser.h"
#include "cudacalls.cuh"
#include "logger.h"
#include "errors.h"

process::process(input* iinput, int exp_idx) {
	process::iinput = iinput;
	process::exp_idx = exp_idx;
	process::h_datacube = NULL;
	process::d_datacube = NULL;
	process::stages = iinput->stages;
}

process::~process() {
	delete process::h_datacube;
	delete process::d_datacube;
}

void process::copyDeviceDatacubeToHost() {
	delete h_datacube;
	process::h_datacube = new hcube(d_datacube);
}

void process::copyHostDatacubeToDevice() {
	delete d_datacube;
	process::d_datacube = new dcube(h_datacube);
}

void process::cropToEvenSquareOnHost() {
	long min_dim = process::iinput->dim[0] < process::iinput->dim[1] ? process::iinput->dim[0] : process::iinput->dim[1];
	if (min_dim % 2 == 1) {
		min_dim--;
	}
	process::h_datacube->crop(rectangle(0, 0, min_dim, min_dim));
}

void process::cropDatacubeToSmallestDimensionSliceOnDevice() {
	std::vector<rectangle> last_regions;
	for (std::vector<dspslice*>::iterator it = process::d_datacube->slices.begin(); it != process::d_datacube->slices.end(); ++it) {
		last_regions.push_back((*it)->region);
	}
	d_datacube->crop(d_datacube->getSmallestSliceRegion());
	process::last_crop_regions = last_regions;
}

void process::fitPolyToSpaxelAndSubtractOnDevice(int poly_order) {
	// Check cube passes integrity check.
	if (d_datacube->state != OK) {
		throw_error(CPROCESS_FIT_AND_SUBTRACT_POLY_FAIL_INTEGRITY_CHECK);
	}

	// Initialise CULA
	culaStatus s;
	s = culaInitialize();
	if (s != culaNoError) {
		throw_error(CULA_INITIALISATION_ERROR);
	}

	// Define some useful variables.
	//
	int n_slices = process::d_datacube->slices.size();
	int n_spaxels = process::d_datacube->getNumberOfSpaxels();
	int n_spaxels_per_slice = n_spaxels / n_slices;

	// Create an array of Complex pointers [p_d_data_slices] on the device with each pointer pointing towards the data
	// for each slice in the datacube. Note that although the data at datacube->slices[i]->p_data is on the device, 
	// the pointer itself resides in memory on the host, so we must use a memcpyhd to copy the pointer to the device.
	//
	Complex** p_d_data_slices = dmemory<Complex*>::malloc(n_slices*sizeof(Complex*), true);
	for (int i = 0; i < n_slices; i++) {
		dmemory<Complex*>::memcpyhd(&p_d_data_slices[i], &process::d_datacube->slices[i]->p_data, sizeof(Complex*));
	}

	// Create a 2D array [p_d_data_spaxels] of dimensions [n_spaxels][n_slices] to contain the spaxel data. 
	// We must make the pointer array on the host and then malloc [n_slices] worth of memory on the device 
	// for each pointer otherwise, as above, we end up trying to access a pointer on the device from the host. 
	// Once this is done we can then copy the pointer array to device memory.
	//
	Complex** p_d_d_data_spaxels;	// array of pointers (ON DEVICE) to spaxel data memory (ON DEVICE)
	Complex** p_h_d_data_spaxels;	// array of pointers (ON HOST) to spaxel data memory (ON DEVICE)

	p_h_d_data_spaxels = hmemory<Complex*>::malloc(n_spaxels_per_slice*sizeof(Complex*), true);
	for (int i = 0; i < n_spaxels_per_slice; i++) {
		p_h_d_data_spaxels[i] = dmemory<Complex>::malloc(n_slices*sizeof(Complex), true);
	}
	p_d_d_data_spaxels = dmemory<Complex*>::malloc(n_spaxels_per_slice*sizeof(Complex*), true);
	memory<Complex*>::memcpyhd(p_d_d_data_spaxels, p_h_d_data_spaxels, n_spaxels_per_slice*sizeof(Complex*));

	// Call the CUDA function to populate [p_d_data_spaxels].
	//
	if (cudaGetSpaxelData2D(stoi(process::iinput->config_device["nCUDABLOCKS"]),
		stoi(process::iinput->config_device["nCUDATHREADSPERBLOCK"]),
		p_d_data_slices, p_d_d_data_spaxels, n_slices, n_spaxels_per_slice) != cudaSuccess) {
		throw_error(CUDA_FAIL_GET_SPAXEL_DATA_2D);
	}
	if (cudaThreadSynchronize() != cudaSuccess) {
		throw_error(CUDA_FAIL_SYNCHRONIZE);
	}

	// Create a 2D array [p_d_data_spaxels_bitmask] of dimensions [n_spaxels][n_slices] to contain the 
	// bit masked spaxel data (i.e. data > 0 == 1). 
	//
	int** p_d_d_data_spaxels_bitmask;	// array of pointers (ON DEVICE) to spaxel data bitmask memory (ON DEVICE)
	int** p_h_d_data_spaxels_bitmask;	// array of pointers (ON HOST) to spaxel data bitmask memory (ON DEVICE)

	p_h_d_data_spaxels_bitmask = hmemory<int*>::malloc(n_spaxels_per_slice*sizeof(int*), true);
	for (int i = 0; i < n_spaxels_per_slice; i++) {
		p_h_d_data_spaxels_bitmask[i] = dmemory<int>::malloc(n_slices*sizeof(int), true);
	}
	p_d_d_data_spaxels_bitmask = dmemory<int*>::malloc(n_spaxels_per_slice*sizeof(int*), true);
	memory<int*>::memcpyhd(p_d_d_data_spaxels_bitmask, p_h_d_data_spaxels_bitmask, n_spaxels_per_slice*sizeof(int*));

	// Call the CUDA function to populate [p_d_data_spaxels_bitmask].
	//
	if (cudaMakeBitmask2D(stoi(process::iinput->config_device["nCUDABLOCKS"]),
		stoi(process::iinput->config_device["nCUDATHREADSPERBLOCK"]),
		p_d_d_data_spaxels, p_d_d_data_spaxels_bitmask, n_slices, n_spaxels_per_slice) != cudaSuccess) {
		throw_error(CUDA_FAIL_MAKE_BITMASK_2D);
	}
	if (cudaThreadSynchronize() != cudaSuccess) {
		throw_error(CUDA_FAIL_SYNCHRONIZE);
	}

	// LEAST SQUARES
	// 
	// We need to form the matrices required for least squares, i.e. minimising | Ax - B |, where:
	// 
	// A = [ 1 x x^2 .. x^n ] , x = [ a(0) ] and B = [ y0 ] 
	//     [ 1 x x^2 .. x^n ]       [ a(1) ]         [ y1 ]
	//     [ . . .   .. .   ]       [ a(n) ]         [ .  ]
    //     [ . . .   .. .   ]                        [ .  ]
	//
	// i.e. A is an (n+1) X m matrix, where m is the number of data points and n is the polynomial order, 
	//      x is a nrhs X (n+1) matrix, and 
	//      B is a nrhs X m matrix.
	//
	// If we were only solving for a single series of data points, nrhs would be 1. Solving for multiple 
	// series at the same time can be done by populating columns in the x and B arrays. Note that the CULA 
	// routines expect A and B to be represented in column-major format.
	//
	// A caveat to solving multiple series simultaneously is that we must "batch" spaxels with the same 
	// matrix A (i.e. they have data points populated at the same wavelengths).
	// 
    // To do this, we consider each spaxel in turn and compare it to the others using the bitmask. If we 
	// find any the same, we batch the spaxels together and do our LSQ calculation with the nhrs = number 
	// of similar bitmasks found. This significantly decreases the overhead compared to calculating each 
	// spaxel separately.
	//
	// First we create an array to hold the spaxel polynomial coefficients.
	//
	Complex** p_h_d_spaxel_poly_coeffs;		// array of pointers (ON DEVICE) to spaxel coeffs memory (ON DEVICE)
	Complex** p_d_d_spaxel_poly_coeffs;		// array of pointers (ON HOST) to spaxel coeffs memory (ON DEVICE)

	p_h_d_spaxel_poly_coeffs = hmemory<Complex*>::malloc(n_spaxels_per_slice*sizeof(Complex*), true);
	for (int i = 0; i < n_spaxels_per_slice; i++) {
		p_h_d_spaxel_poly_coeffs[i] = dmemory<Complex>::malloc((poly_order + 1)*sizeof(Complex), true);
	}
	p_d_d_spaxel_poly_coeffs = dmemory<Complex*>::malloc(n_spaxels_per_slice*sizeof(Complex*), true);
	memory<Complex*>::memcpyhd(p_d_d_spaxel_poly_coeffs, p_h_d_spaxel_poly_coeffs, n_spaxels_per_slice*sizeof(Complex*));

	// And an array to keep track of which spaxels we've processed (as we're batching).
	//
	std::vector<bool> processed_spaxels(n_spaxels_per_slice, false);
	for (int i = 0; i < n_spaxels_per_slice; i++) { 

		// Check this spaxel hasn't been processed already.
		//
		if (processed_spaxels[i] == true) {
			continue;
		}

		// Call the CUDA function to compare the current spaxel's bitmask to the others, this 
		// creates an array [d_truth] containing the indexes of the spaxels that have evaluated 
		// as true to the comparison.
		//
		int* d_truth_bitmask = dmemory<int>::malloc(n_spaxels_per_slice*sizeof(int), true);
		if (cudaCompareArray2D(stoi(process::iinput->config_device["nCUDABLOCKS"]),
			stoi(process::iinput->config_device["nCUDATHREADSPERBLOCK"]),
			p_d_d_data_spaxels_bitmask, d_truth_bitmask, i, n_slices, n_spaxels_per_slice) != cudaSuccess) {
			throw_error(CUDA_FAIL_COMPARE_BITMASK_2D);
		}
		if (cudaThreadSynchronize() != cudaSuccess) {
			throw_error(CUDA_FAIL_SYNCHRONIZE);
		}

		// Copy the truth array to the host.
		//
		int* h_truth_bitmask = hmemory<int>::malloc(n_spaxels_per_slice*sizeof(int), true);
		memory<int>::memcpydh(h_truth_bitmask, d_truth_bitmask, n_spaxels_per_slice*sizeof(int));

		std::vector<int> this_spaxel_similar_bitmask_spaxels_indexes;
		for (int j = 0; j < n_spaxels_per_slice; j++) {
			if (h_truth_bitmask[j] == 1) {
				this_spaxel_similar_bitmask_spaxels_indexes.push_back(j);
			}
		}

		// Get this spaxel's bitmask data and use this to ascertain which slice indexes are to be 
		// considered for LSQ.
		int *this_spaxel_bitmask = hmemory<int>::malloc(n_slices*sizeof(int), true);
		memory<int>::memcpydh(this_spaxel_bitmask, p_h_d_data_spaxels_bitmask[i], n_slices*sizeof(int));

		std::vector<int> this_spaxel_valid_slice_indexes;
		for (int j = 0; j < n_slices; j++) {
			if (this_spaxel_bitmask[j] == 1) {
				this_spaxel_valid_slice_indexes.push_back(j);
			}
		}

		// Find and populate parameters required to construct LSQ matrices.
		//
		// number of elements for this spaxel [this_n_elements_per_spaxel]
		int this_n_elements_per_spaxel = 0;
		for (int j = 0; j < n_slices; j++) {
			if (this_spaxel_bitmask[j] == 1) {
				this_n_elements_per_spaxel++;
			}
		}
		if (this_n_elements_per_spaxel < poly_order + 1) {		// insufficient elements to fit
			continue;
		}

		// number of spaxels with similar bitmask to this spaxel [this_n_similar_bitmasks]
		int this_n_similar_bitmasks = 0;
		for (int j = 0; j < n_spaxels_per_slice; j++) {
			if (h_truth_bitmask[j] == 1) {
				this_n_similar_bitmasks++;
			}
			// Flag this spaxel as processed.
			if (processed_spaxels[j] == false && h_truth_bitmask[j] == 1) {
				processed_spaxels[j] = true;
			}
		}

		char trans = 'N';
		int n = poly_order + 1;
		int m = this_n_elements_per_spaxel;
		int lda = this_n_elements_per_spaxel;
		int ldb = this_n_elements_per_spaxel;
		int nrhs = this_n_similar_bitmasks;

		// Form matrix A in column-major and copy to device.
		//
		Complex* h_A;
		h_A = hmemory<Complex>::malloc(this_n_elements_per_spaxel*(poly_order + 1)*sizeof(Complex), true);
		for (int j = 0; j < poly_order + 1; j++) {
			for (int k = 0; k < this_n_elements_per_spaxel; k++) {
				h_A[k + (j* this_n_elements_per_spaxel)].x = pow(process::iinput->wavelengths[this_spaxel_valid_slice_indexes[k]], j);
				h_A[k + (j* this_n_elements_per_spaxel)].y = 0;
			}
		}
		Complex* d_A;
		d_A = dmemory<Complex>::malloc(this_n_elements_per_spaxel*(poly_order + 1)*sizeof(Complex), true);
		dmemory<Complex>::memcpyhd(d_A, h_A, this_n_elements_per_spaxel*(poly_order + 1)*sizeof(Complex));

		// Form matrix B in column-major on device.
		//
		Complex* d_B;
		d_B = dmemory<Complex>::malloc(this_n_elements_per_spaxel*nrhs*sizeof(Complex), true);
		for (int j = 0; j < nrhs; j++) {
			dmemory<Complex>::memcpydd(&d_B[j*this_n_elements_per_spaxel], p_h_d_data_spaxels[this_spaxel_similar_bitmask_spaxels_indexes[j]], 
				this_n_elements_per_spaxel*sizeof(Complex));
		}

		// Solve!
		//
		culaStatus s;
		s = culaDeviceZgels(trans, m, n, nrhs, reinterpret_cast<culaDeviceDoubleComplex*>(d_A), lda, reinterpret_cast<culaDeviceDoubleComplex*>(d_B), ldb);
		if (s != culaNoError) {
			throw_error(CULA_ZGELS_ERROR);
		}

		// Save off coeffs as 1D array [p_d_spaxel_poly_coeffs] of size ([poly_order] + 1) * [n_spaxels_per_slice], placing
		// the coeffs into the correct place in the array.
		//
		for (int j = 0; j < nrhs; j++) {
			dmemory<Complex>::memcpydd(p_h_d_spaxel_poly_coeffs[this_spaxel_similar_bitmask_spaxels_indexes[j]], &d_B[j*(this_n_elements_per_spaxel)], 
				(poly_order + 1)*sizeof(Complex));
		}

		hmemory<int>::free(h_truth_bitmask);
		dmemory<int>::free(d_truth_bitmask);
		hmemory<int>::free(this_spaxel_bitmask);
		hmemory<Complex>::free(h_A);
		dmemory<Complex>::free(d_A);
		dmemory<Complex>::free(d_B);
	}

	// Evaluate polynomial and subtract.
	//
	int* wavelengths;
	wavelengths = dmemory<int>::malloc(process::iinput->wavelengths.size()*sizeof(int), true);
	dmemory<int>::memcpyhd(wavelengths, &process::iinput->wavelengths[0], process::iinput->wavelengths.size()*sizeof(int));
	cudaPolySub2D(stoi(process::iinput->config_device["nCUDABLOCKS"]), stoi(process::iinput->config_device["nCUDATHREADSPERBLOCK"]),
		p_d_data_slices, p_d_d_data_spaxels_bitmask, p_d_d_spaxel_poly_coeffs, poly_order + 1, wavelengths, n_slices, n_spaxels_per_slice);
	if (cudaThreadSynchronize() != cudaSuccess) {
		throw_error(CUDA_FAIL_SYNCHRONIZE);
	}

	// Free memory and shut down CULA.

	dmemory<Complex*>::free(p_d_data_slices);

	for (int i = 0; i < n_spaxels_per_slice; i++) {
		dmemory<Complex>::free(p_h_d_data_spaxels[i]);
	}
	hmemory<Complex*>::free(p_h_d_data_spaxels);
	dmemory<Complex*>::free(p_d_d_data_spaxels);

	for (int i = 0; i < n_spaxels_per_slice; i++) {
		dmemory<int>::free(p_h_d_data_spaxels_bitmask[i]);
	}
	hmemory<int*>::free(p_h_d_data_spaxels_bitmask);
	dmemory<int*>::free(p_d_d_data_spaxels_bitmask);

	for (int i = 0; i < n_spaxels_per_slice; i++) {
		dmemory<Complex>::free(p_h_d_spaxel_poly_coeffs[i]);
	}
	hmemory<Complex*>::free(p_h_d_spaxel_poly_coeffs);
	dmemory<Complex*>::free(p_d_d_spaxel_poly_coeffs);

	culaShutdown();

}

void process::fftOnDevice() {
	process::d_datacube->fft(false);
	if (cudaThreadSynchronize() != cudaSuccess) {
		throw_error(CUDA_FAIL_SYNCHRONIZE);
	}
	for (std::vector<dspslice*>::iterator it = process::d_datacube->slices.begin(); it != process::d_datacube->slices.end(); ++it) {
		cudaScale2D(stoi(process::iinput->config_device["nCUDABLOCKS"]),
			stoi(process::iinput->config_device["nCUDATHREADSPERBLOCK"]), 
			(*it)->p_data, (1. / ((*it)->getDimensions().x*(*it)->getDimensions().y)), (*it)->memsize / sizeof(Complex));
		if (cudaThreadSynchronize() != cudaSuccess) {
			throw_error(CUDA_FAIL_SYNCHRONIZE);
		}
	}
}

void process::fftshiftOnDevice() {
	dcube* d_datacube_tmp = process::d_datacube->deepcopy();
	d_datacube_tmp->clear();
	for (int i = 0; i < process::d_datacube->slices.size(); i++) {
		Complex* p_data_in = process::d_datacube->slices[i]->p_data;
		Complex* p_data_out = d_datacube_tmp->slices[i]->p_data;
		long x_size = process::d_datacube->slices[i]->region.x_size;
		if (cudaFftShift2D(stoi(process::iinput->config_device["nCUDABLOCKS"]),
			stoi(process::iinput->config_device["nCUDATHREADSPERBLOCK"]),
			p_data_in, p_data_out, x_size) != cudaSuccess) {
			throw_error(CUDA_FAIL_FFTSHIFT);
		}
		if (cudaThreadSynchronize() != cudaSuccess) {
			throw_error(CUDA_FAIL_SYNCHRONIZE);
		}
	}
	delete d_datacube;
	process::d_datacube = d_datacube_tmp;
}

void process::growDatacubeToLargestDimensionSliceOnDevice() {
	std::vector<rectangle> last_regions;
	for (std::vector<dspslice*>::iterator it = process::d_datacube->slices.begin(); it != process::d_datacube->slices.end(); ++it) {
		last_regions.push_back((*it)->region);
	}
	d_datacube->grow(d_datacube->getLargestSliceRegion());
	process::last_grow_regions = last_regions;
}

void process::iFftOnDevice() {
	process::d_datacube->fft(true);
	if (cudaThreadSynchronize() != cudaSuccess) {
		throw_error(CUDA_FAIL_SYNCHRONIZE);
	}
}

void process::iFftshiftOnDevice() {
	dcube* d_datacube_tmp = process::d_datacube->deepcopy();
	d_datacube_tmp->clear();
	for (int i = 0; i < process::d_datacube->slices.size(); i++) {
		Complex* p_data_in = process::d_datacube->slices[i]->p_data;
		Complex* p_data_out = d_datacube_tmp->slices[i]->p_data;
		long x_size = process::d_datacube->slices[i]->region.x_size;
		if (cudaIFftShift2D(stoi(process::iinput->config_device["nCUDABLOCKS"]),
			stoi(process::iinput->config_device["nCUDATHREADSPERBLOCK"]),
			p_data_in, p_data_out, x_size) != cudaSuccess) {
			throw_error(CUDA_FAIL_IFFTSHIFT);
		}
		if (cudaThreadSynchronize() != cudaSuccess) {
			throw_error(CUDA_FAIL_SYNCHRONIZE);
		}
	}
	delete d_datacube;
	process::d_datacube = d_datacube_tmp;
}

void process::revertLastCrop() {
	if (process::last_crop_regions.size() != process::d_datacube->slices.size()) {
		throw_error(CPROCESS_REVERT_CROP_REGIONS_INVALID);
	}
	process::d_datacube->grow(process::last_crop_regions);

	// empty last crop factors
	process::last_crop_regions.clear();
}

void process::revertLastGrow() {
	if (process::last_grow_regions.size() != process::d_datacube->slices.size()) {
		throw_error(CPROCESS_REVERT_GROW_REGIONS_INVALID);
	}
	process::d_datacube->crop(process::last_grow_regions);

	// empty last grow factors
	process::last_grow_regions.clear();
}

void process::revertLastRescale() {
	if (process::last_rescale_regions.size() != process::d_datacube->slices.size()) {
		throw_error(CPROCESS_REVERT_RESCALE_REGIONS_INVALID);
	}
	std::vector<double> inverse_scale_factors;
	for (int i = 0; i < process::d_datacube->slices.size(); i++) {
		inverse_scale_factors.push_back((double)process::last_rescale_regions[i].x_size / (double)d_datacube->slices[i]->region.x_size);
	}

	// need to roll phase (spatial translation) for odd sized frames, otherwise there's a 0.5 pixel offset in x and y compared to the even frames after ifft.
	for (int i = 0; i < process::d_datacube->slices.size(); i++) {
		double2 offset;
		if (inverse_scale_factors[i] < 1 && process::d_datacube->slices[i]->region.x_size % 2 != 0) {
			offset.x = 0.5;
			offset.y = 0.5;
		} else if (inverse_scale_factors[i] > 1 && process::d_datacube->slices[i]->region.x_size % 2 != 0) {
			offset.x = -0.5;
			offset.y = -0.5;
		} else {
			continue;
		}
		long x_size = process::d_datacube->slices[i]->region.x_size;
		if (cudaTranslate2D(stoi(process::iinput->config_device["nCUDABLOCKS"]),
			stoi(process::iinput->config_device["nCUDATHREADSPERBLOCK"]),
			process::d_datacube->slices[i]->p_data, offset, x_size) != cudaSuccess) {
			throw_error(CUDA_FAIL_TRANSLATE_2D);
		}
		if (cudaThreadSynchronize() != cudaSuccess) {
			throw_error(CUDA_FAIL_SYNCHRONIZE);
		}
	}
	process::d_datacube->rescale(inverse_scale_factors);

	// empty last rescale factors
	process::last_rescale_regions.clear();
}

void process::makeDatacubeOnHost() {
	process::h_datacube = process::iinput->makeCube(process::exp_idx, true);
}

void process::rescaleDatacubeToReferenceWavelengthOnDevice(int reference_wavelength) {
	std::vector<double> scale_factors;
	std::vector<rectangle> last_regions;
	for (std::vector<dspslice*>::iterator it = process::d_datacube->slices.begin(); it != process::d_datacube->slices.end(); ++it) {
		last_regions.push_back((*it)->region);
		scale_factors.push_back(reference_wavelength / (double)(*it)->wavelength);
	}
	process::d_datacube->rescale(scale_factors);
	// need to roll phase (spatial translation) for odd sized frames, otherwise there's a 0.5 pixel offset in x and y compared to the even frames after ifft.
	for (int i = 0; i < process::d_datacube->slices.size(); i++) {
		double2 offset;
		if (scale_factors[i] < 1 && process::d_datacube->slices[i]->region.x_size % 2 != 0) {
			offset.x = 0.5;
			offset.y = 0.5;
		} else if (scale_factors[i] > 1 && process::d_datacube->slices[i]->region.x_size % 2 != 0) {
			offset.x = -0.5;
			offset.y = -0.5;
		} else {
			continue;
		}
		long x_size = process::d_datacube->slices[i]->region.x_size;
		if (cudaTranslate2D(stoi(process::iinput->config_device["nCUDABLOCKS"]),
			stoi(process::iinput->config_device["nCUDATHREADSPERBLOCK"]),
			process::d_datacube->slices[i]->p_data, offset, x_size) != cudaSuccess) {
			throw_error(CUDA_FAIL_TRANSLATE_2D);
		}
		if (cudaThreadSynchronize() != cudaSuccess) {
			throw_error(CUDA_FAIL_SYNCHRONIZE);
		}
	}
	process::last_rescale_regions = last_regions;
}

void process::setDataToAmplitude() {
	for (std::vector<dspslice*>::iterator it = process::d_datacube->slices.begin(); it != process::d_datacube->slices.end(); ++it) {
		if (cudaSetComplexRealAsAmplitude2D(stoi(process::iinput->config_device["nCUDABLOCKS"]),
			stoi(process::iinput->config_device["nCUDATHREADSPERBLOCK"]),
			(*it)->p_data, (*it)->memsize / sizeof(Complex)) != cudaSuccess) {
			throw_error(CUDA_FAIL_SET_COMPLEX_REAL_AS_AMPLITUDE);
		}
		if (cudaThreadSynchronize() != cudaSuccess) {
			throw_error(CUDA_FAIL_SYNCHRONIZE);
		}
	}
}

void process::run() {
	/*
	Run a process.
	*/
	long nstages = process::stages.size();
	sprintf(process::message_buffer, "\tPROCESS\t\tstarting new process with process id %d", process::exp_idx);
	to_stdout(process::message_buffer);
	for (int s = 0; s < nstages; s++) {
		process::step(s+1, nstages);
	}
	sprintf(process::message_buffer, "\tPROCESS\t\tprocess %d complete", process::exp_idx);
	to_stdout(process::message_buffer);
}

void process::step(int stage, int nstages) {
	/* 
	Step through process chain by one stage.
	*/
	switch (process::stages.front()) {
	case COPY_DEVICE_DATACUBE_TO_HOST:
		sprintf(process::message_buffer, "%d\tPROCESS (%d/%d)\tcopying device datacube to host", process::exp_idx, stage, nstages);
		to_stdout(process::message_buffer);
		process::copyDeviceDatacubeToHost();
		break;
	case COPY_HOST_DATACUBE_TO_DEVICE:
		sprintf(process::message_buffer, "%d\tPROCESS (%d/%d)\tcopying host datacube to device", process::exp_idx, stage, nstages);
		to_stdout(process::message_buffer);
		process::copyHostDatacubeToDevice();
		break;
	case D_CROP_DATACUBE_TO_SMALLEST_DIMENSION_SLICE:
		sprintf(process::message_buffer, "%d\tPROCESS (%d/%d)\tcropping datacube to smallest dimension slice on device", process::exp_idx, stage, nstages);
		to_stdout(process::message_buffer);
		process::cropDatacubeToSmallestDimensionSliceOnDevice();
		break;
	case D_SPAXEL_FIT_POLY_AND_SUBTRACT:
		sprintf(process::message_buffer, "%d\tPROCESS (%d/%d)\tfitting polynomial to spaxels and subtracting on device", process::exp_idx, stage, nstages);
		to_stdout(process::message_buffer);
		process::fitPolyToSpaxelAndSubtractOnDevice(stoi(process::iinput->stage_parameters[D_SPAXEL_FIT_POLY_AND_SUBTRACT]["POLY_ORDER"]));
		break;
	case D_FFT:
		sprintf(process::message_buffer, "%d\tPROCESS (%d/%d)\tffting datacube on device", process::exp_idx, stage, nstages);
		to_stdout(process::message_buffer);
		process::fftOnDevice();
		break;
	case D_FFTSHIFT:
		sprintf(process::message_buffer, "%d\tPROCESS (%d/%d)\tfftshifting datacube on device", process::exp_idx, stage, nstages);
		to_stdout(process::message_buffer);
		process::fftshiftOnDevice();
		break;
	case D_GROW_DATACUBE_TO_LARGEST_DIMENSION_SLICE:
		sprintf(process::message_buffer, "%d\tPROCESS (%d/%d)\tgrowing datacube to largest dimension slice on device", process::exp_idx, stage, nstages);
		to_stdout(process::message_buffer);
		process::growDatacubeToLargestDimensionSliceOnDevice();
		break;
	case D_IFFT:
		sprintf(process::message_buffer, "%d\tPROCESS (%d/%d)\tiffting datacube on device", process::exp_idx, stage, nstages);
		to_stdout(process::message_buffer);
		process::iFftOnDevice();
		break;
	case D_IFFTSHIFT:
		sprintf(process::message_buffer, "%d\tPROCESS (%d/%d)\tifftshifting datacube on device", process::exp_idx, stage, nstages);
		to_stdout(process::message_buffer);
		process::iFftshiftOnDevice();
		break;
	case D_REVERT_LAST_CROP:
		sprintf(process::message_buffer, "%d\tPROCESS (%d/%d)\treverting last crop on device", process::exp_idx, stage, nstages);
		to_stdout(process::message_buffer);
		process::revertLastCrop();
		break;
	case D_REVERT_LAST_GROW:
		sprintf(process::message_buffer, "%d\tPROCESS (%d/%d)\treverting last grow on device", process::exp_idx, stage, nstages);
		to_stdout(process::message_buffer);
		process::revertLastGrow();
		break;
	case D_REVERT_LAST_RESCALE:
		sprintf(process::message_buffer, "%d\tPROCESS (%d/%d)\treverting last rescale on device", process::exp_idx, stage, nstages);
		to_stdout(process::message_buffer);
		process::revertLastRescale();
		break;
	case D_RESCALE_DATACUBE_TO_REFERENCE_WAVELENGTH:
		sprintf(process::message_buffer, "%d\tPROCESS (%d/%d)\tscaling datacube to reference wavelength on device", process::exp_idx, stage, nstages);
		to_stdout(process::message_buffer);
		process::rescaleDatacubeToReferenceWavelengthOnDevice(stoi(process::iinput->stage_parameters[D_RESCALE_DATACUBE_TO_REFERENCE_WAVELENGTH]["WAVELENGTH"]));
		break;
	case D_SET_DATA_TO_AMPLITUDE: 
		sprintf(process::message_buffer, "%d\tPROCESS (%d/%d)\tsetting datacube data to amplitude on device", process::exp_idx, stage, nstages);
		to_stdout(process::message_buffer);
		process::setDataToAmplitude();
		break;
	case H_CROP_TO_EVEN_SQUARE:
		sprintf(process::message_buffer, "%d\tPROCESS (%d/%d)\tcropping datacube to even square on device", process::exp_idx, stage, nstages);
		to_stdout(process::message_buffer);
		process::cropToEvenSquareOnHost();
		break;
	case MAKE_DATACUBE_ON_HOST:
		sprintf(process::message_buffer, "%d\tPROCESS (%d/%d)\tmaking datacube on host", process::exp_idx, stage, nstages);
		to_stdout(process::message_buffer);
		process::makeDatacubeOnHost();
		break;
	default:
		throw_error(CPROCESS_UNKNOWN_STAGE);
	}
	process::stages.pop_front();
}