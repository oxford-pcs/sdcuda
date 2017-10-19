#include "cprocess.h"

#include "ccube.h"
#include "cinput.h"
#include "cclparser.h"
#include "cudacalls.cuh"
#include "ckernels.h"

#include "logger.h"

process::process(std::list<process_stages> istages, input* iinput, clparser* iclparser, int exp_idx) {
	process::stages = istages;
	process::iinput = iinput;
	process::iclparser = iclparser;
	process::exp_idx = exp_idx;
	process::h_datacube = NULL;
	process::d_datacube = NULL;
}

process::~process() {
	delete h_datacube;
	delete d_datacube;
}

void process::copyDeviceDatacubeToHost() {
	process::h_datacube = new hcube(d_datacube);
}

void process::copyHostDatacubeToDevice() {
	process::d_datacube = new dcube(h_datacube);
}

void process::correctOffsetOnDevice() {
	LanczosShift* lkernel = new LanczosShift(LANCZOS_KERNEL_SIZE, 0.5, 0.5);
	dcube* d_datacube_tmp = process::d_datacube->deepcopy();
	for (int i = 0; i < process::d_datacube->slices.size(); i++) {
		if (process::d_datacube->slices[i]->region.x_size % 2 == 0) {
			continue;
		}
		Complex* p_data_in = process::d_datacube->slices[i]->p_data;
		Complex* p_data_out = d_datacube_tmp->slices[i]->p_data;
		long x_size = process::d_datacube->slices[i]->region.x_size;
		cudaConvolveKernelReal2D(p_data_in, p_data_out, x_size, lkernel->p_kcoeffs, lkernel->ksize);
		if (cudaThreadSynchronize() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}
	}
	delete d_datacube;
	delete lkernel;
	process::d_datacube = d_datacube_tmp;
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
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
	}
}

void process::fftshiftOnDevice() {
	dcube* d_datacube_tmp = process::d_datacube->deepcopy();
	d_datacube_tmp->clear();
	for (int i = 0; i < process::d_datacube->slices.size(); i++) {
		Complex* p_data_in = process::d_datacube->slices[i]->p_data;
		Complex* p_data_out = d_datacube_tmp->slices[i]->p_data;
		long x_size = process::d_datacube->slices[i]->region.x_size;
		cudaFftShift2D(p_data_in, p_data_out, x_size);
		if (cudaThreadSynchronize() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}
	}
	delete d_datacube;
	process::d_datacube = d_datacube_tmp;
}

void process::iFftOnDevice() {
	process::d_datacube->fft(true);
	if (cudaThreadSynchronize() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
	}
}

void process::iFftshiftOnDevice() {
	dcube* d_datacube_tmp = process::d_datacube->deepcopy();
	d_datacube_tmp->clear();
	for (int i = 0; i < process::d_datacube->slices.size(); i++) {
		Complex* p_data_in = process::d_datacube->slices[i]->p_data;
		Complex* p_data_out = d_datacube_tmp->slices[i]->p_data;
		long x_size = process::d_datacube->slices[i]->region.x_size;
		cudaIFftShift2D(p_data_in, p_data_out, x_size);
		if (cudaThreadSynchronize() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}
	}
	delete d_datacube;
	process::d_datacube = d_datacube_tmp;
}

void process::makeDatacubeOnHost() {
	h_datacube = process::iinput->makeCube(process::exp_idx, true);
}

void process::normaliseOnDevice() {
	for (std::vector<dspslice*>::iterator it = process::d_datacube->slices.begin(); it != process::d_datacube->slices.end(); ++it) {
		cudaScale2D((*it)->p_data, (1. / ((*it)->getDimensions().x*(*it)->getDimensions().y)), (*it)->memsize / sizeof(Complex));
		if (cudaThreadSynchronize() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}
	}
}

void process::rescaleOnDevice() {
	process::d_datacube->rescale(process::d_datacube->slices[SLICE_RESCALE_INDEX]->wavelength);
}

void process::setDataToAmplitude() {
	for (std::vector<dspslice*>::iterator it = process::d_datacube->slices.begin(); it != process::d_datacube->slices.end(); ++it) {
		cudaSetComplexRealAsAmplitude((*it)->p_data, (*it)->memsize / sizeof(Complex));
		if (cudaThreadSynchronize() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}
	}
}

void process::run() {
	/*
	Run a process.
	*/
	long nstages = process::stages.size();
	process_to_stdout("starting new process", process::exp_idx);
	for (int s = 0; s < nstages; s++) {
		process::step();
	}
	process_to_stdout("process complete", process::exp_idx);
}

void process::step() {
	/* 
	Step through process chain by one stage.
	*/
	switch (process::stages.front()) {
	case COPY_DEVICE_DATACUBE_TO_HOST:
		process::copyDeviceDatacubeToHost();
		process_to_stdout("copied device datacube to host", process::exp_idx);
		break;
	case COPY_HOST_DATACUBE_TO_DEVICE:
		process::copyHostDatacubeToDevice();
		process_to_stdout("copied host datacube to device", process::exp_idx);
		break;
	case D_CORRECT_OFFSET_ON_DEVICE:
		process::correctOffsetOnDevice();
		process_to_stdout("correcting for offset on device", process::exp_idx);
		break;
	case D_CROP_TO_SMALLEST_DIMENSION:
		process::cropToSmallestDimensionOnDevice();
		process_to_stdout("cropped datacube to smallest dimension on device", process::exp_idx);
		break;
	case D_FFT:
		process::fftOnDevice();
		process_to_stdout("ffted datacube on device", process::exp_idx);
		break;
	case D_FFTSHIFT:
		process::fftshiftOnDevice();
		process_to_stdout("fftshifted datacube on device", process::exp_idx);
		break;
	case D_IFFT:
		process::iFftOnDevice();
		process_to_stdout("iffted datacube on device", process::exp_idx);
		break;
	case D_IFFTSHIFT:
		process::iFftshiftOnDevice();
		process_to_stdout("ifftshifted datacube on device", process::exp_idx);
		break;
	case D_NORMALISE:
		process::normaliseOnDevice();
		process_to_stdout("normalised datacube on device", process::exp_idx);
		break;
	case D_RESCALE:
		process::rescaleOnDevice();
		process_to_stdout("rescaled datacube on device", process::exp_idx);
		break;
	case D_SET_DATA_TO_AMPLITUDE:
		process::setDataToAmplitude();
		process_to_stdout("set datacube data to amplitude on device", process::exp_idx);
		break;
	case H_CROP_TO_EVEN_SQUARE:
		process::cropToEvenSquareOnHost();
		process_to_stdout("cropped datacube to even square on device", process::exp_idx);
		break;
	case MAKE_DATACUBE_ON_HOST:
		process::makeDatacubeOnHost();
		process_to_stdout("made datacube on host", process::exp_idx);
		break;
	default:
		process_to_stderr("unrecognised step requested", process::exp_idx);
		break;
	}
	process::stages.pop_front();
}