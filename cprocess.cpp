#include "cprocess.h"

#include "ccube.h"
#include "cinput.h"
#include "cclparser.h"
#include "cudacalls.cuh"
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
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
	}
	for (std::vector<dspslice*>::iterator it = process::d_datacube->slices.begin(); it != process::d_datacube->slices.end(); ++it) {
		cudaScale2D((*it)->p_data, (1. / ((*it)->getDimensions().x*(*it)->getDimensions().y)), (*it)->memsize / sizeof(Complex));
		if (cudaThreadSynchronize() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
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

void process::iRescaleByWavelengthOnDevice() {
	std::vector<double> scale_factors;
	for (std::vector<dspslice*>::iterator it = process::d_datacube->slices.begin(); it != process::d_datacube->slices.end(); ++it) {
		scale_factors.push_back((*it)->wavelength / RESCALE_WAVELENGTH);
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
		cudaTranslate2D(process::d_datacube->slices[i]->p_data, offset, x_size);
		if (cudaThreadSynchronize() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
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
		scale_factors.push_back(RESCALE_WAVELENGTH / (*it)->wavelength);
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
		cudaTranslate2D(process::d_datacube->slices[i]->p_data, offset, x_size);
		if (cudaThreadSynchronize() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to synchronize\n");
		}
	}
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
	char buf[100]; sprintf(buf, "starting new process with process id %d", process::exp_idx);
	process_to_stdout(buf, process::exp_idx);
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
	case D_IRESCALE:
		process::iRescaleByWavelengthOnDevice();
		process_to_stdout("inverse rescaled datacube by wavelength on device", process::exp_idx);
		break;
	case D_RESCALE:
		process::rescaleByWavelengthOnDevice();
		process_to_stdout("scaled datacube by wavelength on device", process::exp_idx);
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