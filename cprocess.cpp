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

void process::cropToSmallestDimensionOnDevice() {
	d_datacube->crop(d_datacube->getSmallestSliceRegion());
}

void process::fftOnDevice() {
	process::d_datacube->fft(false);
	if (cudaThreadSynchronize() != cudaSuccess) {
		throw_error(CUDA_FAIL_SYNCHRONIZE);
	}
	for (std::vector<dspslice*>::iterator it = process::d_datacube->slices.begin(); it != process::d_datacube->slices.end(); ++it) {
		cudaScale2D(process::iinput->nCUDABLOCKS, process::iinput->nCUDATHREADSPERBLOCK, (*it)->p_data, 
			(1. / ((*it)->getDimensions().x*(*it)->getDimensions().y)), (*it)->memsize / sizeof(Complex));
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
		cudaFftShift2D(process::iinput->nCUDABLOCKS, process::iinput->nCUDATHREADSPERBLOCK, p_data_in, p_data_out, x_size);
		if (cudaThreadSynchronize() != cudaSuccess) {
			throw_error(CUDA_FAIL_SYNCHRONIZE);
		}
	}
	delete d_datacube;
	process::d_datacube = d_datacube_tmp;
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
		cudaIFftShift2D(process::iinput->nCUDABLOCKS, process::iinput->nCUDATHREADSPERBLOCK, p_data_in, p_data_out, x_size);
		if (cudaThreadSynchronize() != cudaSuccess) {
			throw_error(CUDA_FAIL_SYNCHRONIZE);
		}
	}
	delete d_datacube;
	process::d_datacube = d_datacube_tmp;
}

void process::iRescaleByWavelengthOnDevice() {
	std::vector<double> scale_factors;
	for (std::vector<dspslice*>::iterator it = process::d_datacube->slices.begin(); it != process::d_datacube->slices.end(); ++it) {
		scale_factors.push_back((double)(*it)->wavelength / (double)process::iinput->RESCALE_WAVELENGTH);
	}
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
		cudaTranslate2D(process::iinput->nCUDABLOCKS, process::iinput->nCUDATHREADSPERBLOCK, 
			process::d_datacube->slices[i]->p_data, offset, x_size);
		if (cudaThreadSynchronize() != cudaSuccess) {
			throw_error(CUDA_FAIL_SYNCHRONIZE);
		}
	}
	process::d_datacube->rescale(scale_factors);
}

void process::makeDatacubeOnHost() {
	process::h_datacube = process::iinput->makeCube(process::exp_idx, true);
}

void process::rescaleByWavelengthOnDevice() {
	std::vector<double> scale_factors;
	for (std::vector<dspslice*>::iterator it = process::d_datacube->slices.begin(); it != process::d_datacube->slices.end(); ++it) {
		scale_factors.push_back((double)process::iinput->RESCALE_WAVELENGTH / (double)(*it)->wavelength);
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
		cudaTranslate2D(process::iinput->nCUDABLOCKS, process::iinput->nCUDATHREADSPERBLOCK, 
			process::d_datacube->slices[i]->p_data, offset, x_size);
		if (cudaThreadSynchronize() != cudaSuccess) {
			throw_error(CUDA_FAIL_SYNCHRONIZE);
		}
	}
}

void process::setDataToAmplitude() {
	for (std::vector<dspslice*>::iterator it = process::d_datacube->slices.begin(); it != process::d_datacube->slices.end(); ++it) {
		cudaSetComplexRealAsAmplitude(process::iinput->nCUDABLOCKS, process::iinput->nCUDATHREADSPERBLOCK, 
			(*it)->p_data, (*it)->memsize / sizeof(Complex));
		if (cudaThreadSynchronize() != cudaSuccess) {
			throw_error(CUDA_FAIL_SYNCHRONIZE);
		}
	}
}

void process::run() {
	/*
	Run a process.
	*/
	long nstages = process::iinput->stages.size();
	sprintf(process::message_buffer, "\tPROCESS\tstarting new process with process id %d", process::exp_idx);
	to_stdout(process::message_buffer);
	for (int s = 0; s < nstages; s++) {
		process::step();
	}
	sprintf(process::message_buffer, "\tPROCESS\tprocess %d complete", process::exp_idx);
	to_stdout(process::message_buffer);
}

void process::step() {
	/* 
	Step through process chain by one stage.
	*/
	printf("%d\n", process::iinput->stages.front());
	switch (process::iinput->stages.front()) {
	case COPY_DEVICE_DATACUBE_TO_HOST:
		sprintf(process::message_buffer, "%d\tPROCESS\tcopying device datacube to host", process::exp_idx);
		process::copyDeviceDatacubeToHost();
		to_stdout(process::message_buffer);
		break;
	case COPY_HOST_DATACUBE_TO_DEVICE:
		sprintf(process::message_buffer, "%d\tPROCESS\tcopying host datacube to device", process::exp_idx);
		process::copyHostDatacubeToDevice();
		to_stdout(process::message_buffer);
		break;
	case D_CROP_TO_SMALLEST_DIMENSION:
		sprintf(process::message_buffer, "%d\tPROCESS\tcropping datacube to smallest dimension on device", process::exp_idx);
		process::cropToSmallestDimensionOnDevice();
		to_stdout(process::message_buffer);
		break;
	case D_FFT:
		sprintf(process::message_buffer, "%d\tPROCESS\tffting datacube on device", process::exp_idx);
		process::fftOnDevice();
		to_stdout(process::message_buffer);
		break;
	case D_FFTSHIFT:
		sprintf(process::message_buffer, "%d\tPROCESS\tfftshifting datacube on device", process::exp_idx);
		process::fftshiftOnDevice();
		to_stdout(process::message_buffer);
		break;
	case D_IFFT:
		sprintf(process::message_buffer, "%d\tPROCESS\tiffting datacube on device", process::exp_idx);
		process::iFftOnDevice();
		to_stdout(process::message_buffer);
		break;
	case D_IFFTSHIFT:
		sprintf(process::message_buffer, "%d\tPROCESS\tifftshifting datacube on device", process::exp_idx);
		process::iFftshiftOnDevice();
		to_stdout(process::message_buffer);
		break;
	case D_IRESCALE:
		sprintf(process::message_buffer, "%d\tPROCESS\tinverse rescaling datacube by wavelength on device", process::exp_idx);
		process::iRescaleByWavelengthOnDevice();
		to_stdout(process::message_buffer);
		break;
	case D_RESCALE:
		sprintf(process::message_buffer, "%d\tPROCESS\tscaling datacube by wavelength on device", process::exp_idx);
		process::rescaleByWavelengthOnDevice();
		to_stdout(process::message_buffer);
		break;
	case D_SET_DATA_TO_AMPLITUDE:
		sprintf(process::message_buffer, "%d\tPROCESS\tsetting datacube data to amplitude on device", process::exp_idx);
		process::setDataToAmplitude();
		to_stdout(process::message_buffer);
		break;
	case H_CROP_TO_EVEN_SQUARE:
		sprintf(process::message_buffer, "%d\tPROCESS\tcropping datacube to even square on device", process::exp_idx);
		process::cropToEvenSquareOnHost();
		to_stdout(process::message_buffer);
		break;
	case MAKE_DATACUBE_ON_HOST:
		sprintf(process::message_buffer, "%d\tPROCESS\tmaking datacube on host", process::exp_idx);
		process::makeDatacubeOnHost();
		to_stdout(process::message_buffer);
		break;
	default:
		sprintf(process::message_buffer, "%d\tPROCESS\tcopied host datacube to device", process::exp_idx);
		throw_error(CPROCESS_UNKNOWN_STAGE);
	}
	process::iinput->stages.pop_front();
}