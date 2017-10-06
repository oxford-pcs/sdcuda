#include <cuda_runtime.h>
#include <cula.h>
#include <future>
#include <cufft.h>
#include "cublas_v2.h"
#include "getopt.h"
#include <thread>
#include <chrono>

#include "config.h"

#include "cclparser.h"
#include "cinput.h"
#include "ccube.h"
#include "ccomplex.cuh"
#include "regions.h"
#include "ckernels.h"
#include "cspaxel.h"
#include "cprocess_monitor.h"

hcube* process(input* input, clparser* clparser, int slice_idx) {
	process_monitor pm(PROCESS_STARTED);

	// Make a datacube in host memory
	// 
	hcube* h_datacube = input->makeCube(slice_idx, true);
	pm.next();

	// Crop each slice to an even square
	//
	long min_dim = input->dim[0] < input->dim[1] ? input->dim[0] : input->dim[1];
	if (min_dim % 2 == 1) {
		min_dim--;
	}
	std::vector<rectangle> crop_regions;
	for (int i = 0; i < h_datacube->slices.size(); i++) {
		crop_regions.push_back(rectangle(0, 0, min_dim, min_dim));
	}
	h_datacube->crop(crop_regions);
	pm.next();

	// Copy the host data over to device memory
	//
	dcube* d_datacube = new dcube(h_datacube);
	pm.next();

	// (CUDA) fft (in-place)
	//
	d_datacube->fft(false);
	if (cudaThreadSynchronize() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
	}
	pm.next();

	// (CUDA) normalise (in-place)
	//
	for (std::vector<dspslice*>::iterator it = d_datacube->slices.begin(); it != d_datacube->slices.end(); ++it) {
		cScale2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >((*it)->p_data, (1. / ((*it)->getDimensions().x*(*it)->getDimensions().y)), (*it)->memsize / sizeof(Complex));
		if (cudaThreadSynchronize() != cudaSuccess){
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}
	}
	pm.next();

	// (CUDA) fftshift (out-of-place)
	//
	dcube* d_datacube_tmp = d_datacube->deepcopy();
	d_datacube_tmp->clear();
	for (int i = 0; i < d_datacube->slices.size(); i++) {
		Complex* p_data_in = d_datacube->slices[i]->p_data;
		Complex* p_data_out = d_datacube_tmp->slices[i]->p_data;
		long x_size = d_datacube->slices[i]->region.x_size;
		long y_size = d_datacube->slices[i]->region.y_size;
		cFftShift2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(p_data_in, p_data_out, x_size);
		if (cudaThreadSynchronize() != cudaSuccess){
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}
	}

	delete d_datacube;
	d_datacube = d_datacube_tmp;
	pm.next();

	// Rescale the device datacube by trimming in frequency space
	//
	d_datacube->rescale(d_datacube->slices[SLICE_RESCALE_INDEX]->wavelength);
	pm.next();

	// (CUDA) ifftshift (out-of-place)
	//
	d_datacube_tmp = d_datacube->deepcopy();
	d_datacube_tmp->clear();
	for (int i = 0; i < d_datacube->slices.size(); i++) {
		Complex* p_data_in = d_datacube->slices[i]->p_data;
		Complex* p_data_out = d_datacube_tmp->slices[i]->p_data;
		long x_size = d_datacube->slices[i]->region.x_size;
		long y_size = d_datacube->slices[i]->region.y_size;
		printf("%d\n", x_size);
		cIFftShift2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(p_data_in, p_data_out, x_size);
		if (cudaThreadSynchronize() != cudaSuccess){
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}
	}
	delete d_datacube;
	d_datacube = d_datacube_tmp;
	pm.next();

	/*


	// (CUDA) ifft (in-place)
	//
	d_datacube->fft(true);
	if (cudaThreadSynchronize() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
	}
	pm.next();

	*/

	// Crop cube in spatial domain to smallest dimension slice
	//
	std::vector<rectangle> pre_rescale_regions;
	for (std::vector<dspslice*>::iterator it = d_datacube->slices.begin(); it != d_datacube->slices.end(); ++it) {
		pre_rescale_regions.push_back((*it)->region);
	}
	d_datacube->crop(d_datacube->getSmallestSliceRegion());
	pm.next();

	/*

	// As the fft/ifft and shifting operations cause odd sampled images to be centrally offset from even ones by 
	// half a pixel in both x and y, we need to "move" these to align them
	//

	// first we set the cube data to be amplitude only
	for (std::vector<dspslice*>::iterator it = d_datacube->slices.begin(); it != d_datacube->slices.end(); ++it) {
		cSetComplexRealAsAmplitude << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >((*it)->p_data, (*it)->memsize / sizeof(Complex));
		if (cudaThreadSynchronize() != cudaSuccess){
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}
	}

	// then we apply a Lanczos filter
	LanczosShift* kernel = new LanczosShift(LANCZOS_FILTER_SIZE, -0.5, -0.5);
	d_datacube_tmp = d_datacube->deepcopy();
	for (int i = 0; i < d_datacube->slices.size(); i++) {
		if (pre_rescale_regions[i].x_size % 2 == 0) {
			continue;
		}
		Complex* p_data_in = d_datacube->slices[i]->p_data;
		Complex* p_data_out = d_datacube_tmp->slices[i]->p_data;
		long x_size = d_datacube->slices[i]->region.x_size;
		long y_size = d_datacube->slices[i]->region.y_size;
		cConvolveKernelReal2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(p_data_in, p_data_out, x_size, kernel->p_kcoeffs, kernel->ksize);
		if (cudaThreadSynchronize() != cudaSuccess){
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}
	}
	delete d_datacube;
	d_datacube = d_datacube_tmp;
	delete kernel;
	pm.next();

	*/

	// Move datacube back to host and return
	// 
	delete h_datacube;
	h_datacube = new hcube(d_datacube);
	delete d_datacube;

	return h_datacube;
}

int main(int argc, char **argv) {

	// Parse the command line input
	//
	clparser* i_clparser = new clparser(argc, argv);
	if (i_clparser->state != CCLPARSER_OK) {
		exit(EXIT_FAILURE);
	}

	printf("\n");
	// Process the input files parsed from the command line input
	//
	input* i_input = new input(i_clparser->in_FITS_filename, i_clparser->in_params_filename, true);
	if (i_input->state != CINPUT_OK) {
		exit(EXIT_FAILURE);
	}

	printf("\nSending processes for asynchronous reduction...\n\n");
	// Execute asynchronously
	//
	std::vector<std::future<hcube*>> running_processes;
	for (int i = 0; i < i_input->dim[2]; i++) {
		int available_slots = nCPUCORES;
		for (std::vector<std::future<hcube*>>::iterator it = running_processes.begin(); it != running_processes.end(); ++it) {
			if (it->wait_for(std::chrono::microseconds(1)) != future_status::ready) {
				available_slots--;
			}
		}
		printf("MSG:\ti have %d available slot(s)\n", available_slots);
		if (available_slots > 0) {
			printf("MSG:\tmaking cube for exposure number %d\n", i);
			running_processes.push_back(std::async(process, i_input, i_clparser, i));
		} else {
			printf("MSG:\twaiting for next available slot\n");
			bool slot_is_available = false;
			while (!slot_is_available) {
				for (std::vector<std::future<hcube*>>::iterator it = running_processes.begin(); it != running_processes.end(); ++it) {
					if (it->wait_for(std::chrono::milliseconds(1000)) == future_status::ready) {
						printf("MSG:\ti have a new slot available\n");
						running_processes.erase(it);
						slot_is_available = true;
						break;
					}
				}
			}
		}
	}
	// make sure last processes complete
	for (std::vector<std::future<hcube*>>::iterator it = running_processes.begin(); it != running_processes.end(); it++) {
		it->get()->write(AMPLITUDE, i_clparser->out_FITS_filename, true);		// FIXME: need to construct 4d cube!
	}

	delete i_input;
	delete i_clparser;
	
	exit(EXIT_SUCCESS);
}
