#include <cuda_runtime.h>
#include <cula.h>

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
#include "cspaxel.h"

int main(int argc, char **argv) {

	// Parse the command line input.
	cclparser clparser = cclparser(argc, argv);
	if (clparser.state != CCLPARSER_OK) {
		exit(0);
	}

	// Process the input files parsed from the command line input.
	input input(clparser.in_FITS_filename, clparser.in_params_filename, true);
	if (input.state != CINPUT_OK) {
		exit(0);
	}

	for (int i = 0; i<input.dim[2]; i++) {				// Then begin, for each exposure in the cube..

		// 1. Make a datacube in host memory.
		// 
		hcube* h_datacube = input.makeCube(i, true);
		
		// 2. Crop each slice to a square.
		//
		long min_dim;
		if (h_datacube->dim[0] >= h_datacube->dim[1]) {
			min_dim = h_datacube->dim[1];
		}
		else {
			min_dim = h_datacube->dim[0];
		}
		if (min_dim % 2 == 1) {
			min_dim--;
		}
		std::vector<rectangle> crop_regions;
		for (int i = 0; i < h_datacube->slices.size(); i++) {
			crop_regions.push_back(rectangle(0, 0, min_dim, min_dim));
		}
		h_datacube->crop(crop_regions);

		// need to manually reset the dimensions [dim[01?]] and number of elements [n_elements] for the datacube
		h_datacube->dim[0] = min_dim;
		h_datacube->dim[1] = min_dim;
		h_datacube->n_elements = h_datacube->dim[0] * h_datacube->dim[1] * h_datacube->dim[2];

		// 3. Copy the host data over to device memory.
		//
		dcube* d_datacube = new dcube(h_datacube);		// copy host data over to device

		// 4. (CUDA) fft and normalise (in-place)
		//
		d_datacube->fft(false); 
		if (cudaThreadSynchronize() != cudaSuccess){
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}
		cScale2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(d_datacube->p_data, (1. / (d_datacube->dim[0] * d_datacube->dim[1])), d_datacube->memsize/sizeof(Complex));
		if (cudaThreadSynchronize() != cudaSuccess){
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}

		// 5. (CUDA) fftshift (out-of-place)
		//
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

		// 6. Rescale the device datacube by trimming in frequency space. Note that this makes the 
		// [n_elements] and [dim] NULL temporarily.
		//
		rectangle rect;
		d_datacube->rescale(d_datacube->wavelengths[0], rect);

		// 7. (CUDA) ifftshift (out-of-place)
		//
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

		// 8. (CUDA) ifft (in-place)
		//
		d_datacube->fft(true);
		if (cudaThreadSynchronize() != cudaSuccess){
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}

		// 9. Crop cube in spatial domain to smallest dimension slice (slices are currently inconsistent sizes) and 
		//    assign [n_elements] and [dim] values to the cube.
		//
		crop_regions.clear();
		std::vector<long> pre_crop_sizes;
		for (std::vector<dspslice>::iterator it = d_datacube->slices.begin(); it != d_datacube->slices.end(); it++) {
			pre_crop_sizes.push_back(it->region.x_size);
			rectangle subregion = rect - it->region;
			crop_regions.push_back(subregion);
		}
		d_datacube->crop(crop_regions);
		d_datacube->dim[0] = rect.x_size;
		d_datacube->dim[1] = rect.y_size;
		d_datacube->n_elements = d_datacube->dim[0] * d_datacube->dim[1] * d_datacube->dim[2];

		// 10. As the fft/ifft and shifting operations cause odd/even sampled images to be centrally offset by 
		//     half a pixel in both x and y, so we need to interpolate the flux to align the images 
		//

		// first we set the cube data to be amplitude only
		cSetComplexRealAsAmplitude << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(d_datacube->p_data, d_datacube->memsize / sizeof(Complex));
		if (cudaThreadSynchronize() != cudaSuccess){
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}

		// then we apply a Lanczos filter
		LanczosShift* kernel = new LanczosShift(LANCZOS_FILTER_SIZE, -0.5, -0.5);
		d_datacube_tmp = d_datacube->copy();
		for (int i = 0; i < d_datacube->slices.size(); i++) {
			if (pre_crop_sizes[i] % 2 == 0) {
				continue;
			}
			Complex* p_data_in = d_datacube->slices[i].p_data;
			Complex* p_data_out = d_datacube_tmp->slices[i].p_data;
			long x_size = d_datacube->slices[i].region.x_size;
			long y_size = d_datacube->slices[i].region.y_size;
			cConvolveKernelReal2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(p_data_in, p_data_out, x_size, kernel->p_kcoeffs, kernel->ksize);
		}
		if (cudaThreadSynchronize() != cudaSuccess){
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}
		delete d_datacube;
		d_datacube = d_datacube_tmp;
		delete kernel;

		
		// ***
		// ** WORKING AREA
		
		/*std::vector<dspaxel> spaxels;
		for (long jj = 0; jj < d_datacube->dim[1]; jj++) {
			for (long jj = 0; jj < d_datacube->dim[1]; jj++) {
				dspaxel spaxel = dspaxel(d_datacube, std::vector<long> {0, 0});
				spaxels.push_back(spaxel);
			}
		}*/

		d_datacube_tmp = d_datacube->copy();
		culaStatus status;
		culaInitialize();
		for (int i = 0; i < d_datacube->slices.size(); i++) {
			status = culaDeviceZgeTransposeInplace(d_datacube_tmp->dim[0], reinterpret_cast<culaDeviceDoubleComplex*>(d_datacube_tmp->slices[i].p_data), 
			d_datacube_tmp->dim[0]);
		}
		printf("%d\n", status);
		printf("Shutting down CULA\n");
		culaShutdown();
		delete d_datacube;
		d_datacube = d_datacube_tmp;
		
		//for (std::vector<dspaxel>::iterator it = spaxels.begin(); it != spaxels.end(); it++) {
		//}
		// **

		// 11. Move datacube back to host.
		// 
		hcube* h_datacube_fft = new hcube(d_datacube);
		h_datacube_fft->write(AMPLITUDE, clparser.out_FITS_filename, true);

		delete h_datacube;
		delete d_datacube;
		delete h_datacube_fft;

	}
}
