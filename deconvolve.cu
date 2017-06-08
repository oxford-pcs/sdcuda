#include <cuda_runtime.h>
#include <cufft.h>

#include "cinput.h"
#include "ccube.h"
#include "ccomplex.cuh"
#include "regions.h"
#include "ckernels.h"

const int nCUDABLOCKS = 32;
const int nCUDATHREADSPERBLOCK = 256;

const int LANCZOS_FILTER_SIZE = 3;

inline bool is_input_ok(int argc, char **argv) {
	if (argc != 4) {
		return false;
	} else {
	}
	return true;
}

int main(int argc, char **argv) {
	if (!is_input_ok(argc, argv)) {
		printf("Input check failed.\n");
		return 1;
	}

	std::string in_fits_filename = argv[1];
	std::string in_params_filename = argv[2];
	std::string out_fits_filename = argv[3];

	input input(in_fits_filename, in_params_filename, true);

	long n_exposures = input.dim[2];
	for (int i = 0; i < n_exposures; i++) {
		// make cube on host
		hcube* h_datacube = input.makeCube(i, true);
		std::vector<rectangle> crop_regions;

		// crop to square, even sided. this is required for fftshifting
		long min_dim;
		if (h_datacube->dim[0] >= h_datacube->dim[1]) {
			min_dim = h_datacube->dim[1];
		} else {
			min_dim = h_datacube->dim[0];
		}
		if (min_dim % 2 == 1) {
			min_dim--;
		}
		for (int i = 0; i < h_datacube->slices.size(); i++) {
			crop_regions.push_back(rectangle(0, 0, min_dim, min_dim));
		}
		h_datacube->crop(crop_regions);
		h_datacube->dim[0] = min_dim;
		h_datacube->dim[1] = min_dim;
		h_datacube->n_elements = h_datacube->dim[0] * h_datacube->dim[1] * h_datacube->dim[2];

		// copy host data over to device
		dcube* d_datacube = new dcube(h_datacube);

		// (CUDA) fft and normalise (in-place)
		d_datacube->fft(false); 
		if (cudaThreadSynchronize() != cudaSuccess){
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}
		cScalePointwise << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(d_datacube->p_data, (1. / (d_datacube->dim[0] * d_datacube->dim[1])), d_datacube->n_elements);
		if (cudaThreadSynchronize() != cudaSuccess){
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}

		// (CUDA) fftshift (out-of-place)
		dcube* d_datacube_tmp = d_datacube->copy();
		d_datacube_tmp->clear();
		for (int i = 0; i < d_datacube->slices.size(); i++) {
			Complex* p_data_in = d_datacube->slices[i].p_data;
			Complex* p_data_out = d_datacube_tmp->slices[i].p_data;
			long x_size = d_datacube->slices[i].region.x_size;
			long y_size = d_datacube->slices[i].region.y_size;
			cFftShift2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(p_data_in, p_data_out, x_size);
		}
		if (cudaThreadSynchronize() != cudaSuccess){
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}
		delete d_datacube;
		d_datacube = d_datacube_tmp;

		// rescale by trimming in frequency space
		rectangle rect = d_datacube->rescale(input.wavelengths[0]);

		// (CUDA) ifftshift (out-of-place)
		d_datacube_tmp = d_datacube->copy();
		d_datacube_tmp->clear();
		for (int i = 0; i < d_datacube->slices.size(); i++) {
			Complex* p_data_in = d_datacube->slices[i].p_data;
			Complex* p_data_out = d_datacube_tmp->slices[i].p_data;
			long x_size = d_datacube->slices[i].region.x_size;
			long y_size = d_datacube->slices[i].region.y_size;
			cIFftShift2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(p_data_in, p_data_out, x_size);
		}
		if (cudaThreadSynchronize() != cudaSuccess){
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}
		delete d_datacube;
		d_datacube = d_datacube_tmp;

		// (CUDA) ifft (in-place)
		d_datacube->fft(true);
		if (cudaThreadSynchronize() != cudaSuccess){
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}

		// crop cube to smallest dimension
		crop_regions.clear();
		std::vector<long> pre_shrink_sizes;
		for (std::vector<spslice>::iterator it = d_datacube->slices.begin(); it != d_datacube->slices.end(); it++) {
			pre_shrink_sizes.push_back(it->region.x_size);
			rectangle subregion = rect - it->region;
			crop_regions.push_back(subregion);
		}
		d_datacube->crop(crop_regions);
		d_datacube->dim[0] = rect.x_size;
		d_datacube->dim[1] = rect.y_size;
		d_datacube->n_elements = d_datacube->dim[0] * d_datacube->dim[1] * h_datacube->dim[2];

		// fft/ifft and shifting operations cause odd/even sampled images to be centrally offset by 
		// half a pixel in both x and y, so we need to interpolate the flux to align the images
		//
		// first we set the cube data to be amplitude only
		//
		cSetComplexRealAsAmplitude << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(d_datacube->p_data, d_datacube->memsize / sizeof(Complex));
		if (cudaThreadSynchronize() != cudaSuccess){
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}
		//
		// then we apply a Lanczos filter
		// 
		LanczosShift* kernel = new LanczosShift(LANCZOS_FILTER_SIZE, -0.5, -0.5);
		d_datacube_tmp = d_datacube->copy();
		for (int i = 0; i < d_datacube->slices.size(); i++) {
			if (pre_shrink_sizes[i] % 2 == 0) {
				continue;
			}
			Complex* p_data_in = d_datacube->slices[i].p_data;
			Complex* p_data_out = d_datacube_tmp->slices[i].p_data;
			long x_size = d_datacube->slices[i].region.x_size;
			long y_size = d_datacube->slices[i].region.y_size;
			cConvolveKernelPointwise << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(p_data_in, p_data_out, x_size, kernel->p_kcoeffs, kernel->ksize);
		}
		if (cudaThreadSynchronize() != cudaSuccess){
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}
		delete d_datacube;
		d_datacube = d_datacube_tmp;
		delete kernel;

		//TODO: CROP OFF EDGES!

		// move back to host
		hcube* h_datacube_fft = new hcube(d_datacube);

		h_datacube_fft->write(AMPLITUDE, out_fits_filename, true);

		delete h_datacube;
		delete d_datacube;
		delete h_datacube_fft;
	}
}
