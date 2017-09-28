#include <cuda_runtime.h>
#include <cufft.h>
#include "cublas_v2.h"
#include "getopt.h"

#include "config.h"

#include "cclparser.h"
#include "cinput.h"
#include "ccube.h"
#include "ccomplex.cuh"
#include "regions.h"
#include "ckernels.h"

int main(int argc, char **argv) {
	// parse command line input
	cclparser clparser = cclparser(argc, argv);
	if (clparser.state != CCLPARSER_OK) {
		exit(0);
	}

	// process input files 
	input input(clparser.in_FITS_filename, clparser.in_params_filename, true);
	if (input.state != CINPUT_OK) {
		exit(0);
	}

	// loop through each exposure
	for (int i = 0; i<input.dim[2]; i++) {
		hcube* h_datacube = input.makeCube(i, true);	// make cube on host
		std::vector<rectangle> crop_regions;			// variable to keep 

		// crop each exposure to a square, even sided with dimensions corresponding to the smallest side.
		// (this is required for fftshifting)
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
		// need to manually reset the dimensions [dim[01?]] and number of elements [n_elements] for the datacube 
		// as the datacube crop function sets these to NULL (the operation is in itself "unsafe", in the sense that 
		// it is possible to make the cube inconsistent by ending up with slices of differing sizes).
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
		cScalePointwise << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(d_datacube->p_data, (1. / (d_datacube->dim[0] * d_datacube->dim[1])), d_datacube->memsize/sizeof(Complex));
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
		for (std::vector<dspslice>::iterator it = d_datacube->slices.begin(); it != d_datacube->slices.end(); it++) {
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

		// move back to host
		hcube* h_datacube_fft = new hcube(d_datacube);

		h_datacube_fft->write(AMPLITUDE, clparser.out_FITS_filename, true);

		delete h_datacube;
		delete d_datacube;
		delete h_datacube_fft;
	}
}
