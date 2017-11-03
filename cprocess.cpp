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

std::vector<rectangle> process::cropDatacubeToSmallestDimensionSliceOnDevice() {
	std::vector<rectangle> last_regions;
	for (std::vector<dspslice*>::iterator it = process::d_datacube->slices.begin(); it != process::d_datacube->slices.end(); ++it) {
		last_regions.push_back((*it)->region);
	}
	d_datacube->crop(d_datacube->getSmallestSliceRegion());
	return last_regions;
}

void process::fitPolyToSpaxelAndSubtractOnDevice(int poly_order) {
	Complex** p_data_slices;	
	p_data_slices = dmemory<Complex*>::malloc(process::d_datacube->slices.size()*sizeof(Complex*), true);
	for (int i = 0; i < process::d_datacube->slices.size(); i++) {
		dmemory<Complex*>::memcpyhd(&p_data_slices[i], &process::d_datacube->slices[i]->p_data, sizeof(Complex*));
	}

	Complex* p_data_spaxels;
	p_data_spaxels = dmemory<Complex>::malloc(process::d_datacube->slices[0]->getNumberOfElements()*
		process::d_datacube->slices.size()*sizeof(Complex), true);

	if (cudaGetSpaxelData2D(stoi(process::iinput->config_device["nCUDABLOCKS"]),
		stoi(process::iinput->config_device["nCUDATHREADSPERBLOCK"]),
		p_data_slices, p_data_spaxels, process::d_datacube->slices.size(),
		process::d_datacube->slices[0]->region.x_size*process::d_datacube->slices[0]->region.y_size) != cudaSuccess) {
		throw_error(CUDA_FAIL_GET_SPAXEL_DATA_2D);
	}
	if (cudaThreadSynchronize() != cudaSuccess) {
		throw_error(CUDA_FAIL_SYNCHRONIZE);
	}

	// form A in column-major and copy to device
	Complex* A;
	A = hmemory<Complex>::malloc(d_datacube->slices.size()*(poly_order + 1)*sizeof(Complex), true);
	for (int j = 0; j < poly_order + 1; j++) {
		for (int i = 0; i < process::d_datacube->slices.size(); i++) {
			A[i + (j* process::d_datacube->slices.size())].x = pow(process::iinput->wavelengths[i], j);
			A[i + (j* process::d_datacube->slices.size())].y = 0;
		}
	}

	Complex* d_A;
	d_A = dmemory<Complex>::malloc(process::d_datacube->slices.size()*(poly_order + 1)*sizeof(Complex), true);
	dmemory<Complex>::memcpyhd(d_A, A, process::d_datacube->slices.size()*(poly_order + 1)*sizeof(Complex));

	// LSQ
	Complex* p_data_spaxel_coeffs;
	Complex* p_spaxel;
	Complex* this_d_A;
	p_spaxel = dmemory<Complex>::malloc(process::d_datacube->slices.size()*sizeof(Complex), true);
	culaStatus s;
	s = culaInitialize();
	if (s != culaNoError) {
	} else {
		p_data_spaxel_coeffs = dmemory<Complex>::malloc(process::d_datacube->slices[0]->getNumberOfElements()*(poly_order + 1)*sizeof(Complex), true);
		for (int i = 0; i < process::d_datacube->slices[0]->getNumberOfElements(); i++) {
			dmemory<Complex>::memcpydd(p_spaxel, &p_data_spaxels[i*process::d_datacube->slices.size()],
				process::d_datacube->slices.size()*sizeof(Complex));

	
			this_d_A = dmemory<Complex>::malloc(process::d_datacube->slices.size()*(poly_order + 1)*sizeof(Complex), true);
			dmemory<Complex>::memcpydd(this_d_A, d_A, process::d_datacube->slices.size()*(poly_order + 1)*sizeof(Complex));

			s = culaDeviceZgels('N', process::d_datacube->slices.size(), (poly_order + 1), 1, reinterpret_cast<culaDeviceDoubleComplex*>(this_d_A),
				process::d_datacube->slices.size(), reinterpret_cast<culaDeviceDoubleComplex*>(p_spaxel),
				process::d_datacube->slices.size());

			dmemory<Complex>::memcpydd(&p_data_spaxel_coeffs[i*(poly_order + 1)], p_spaxel, (poly_order + 1)*sizeof(Complex));

			if (s != culaNoError) {
			} else {

			}
		}
	}

	Complex* p_h_data_spaxel_coeffs = hmemory<Complex>::malloc(process::d_datacube->slices[0]->getNumberOfElements()*(poly_order + 1)*sizeof(Complex), true);
	dmemory<Complex>::memcpydh(p_h_data_spaxel_coeffs, p_data_spaxel_coeffs, 
		process::d_datacube->slices[0]->getNumberOfElements()*(poly_order + 1)*sizeof(Complex));


	// Evaluate polynomial and subtract
	int* wavelengths;
	wavelengths = dmemory<int>::malloc(process::iinput->wavelengths.size()*sizeof(int), true);
	dmemory<int>::memcpyhd(wavelengths, &process::iinput->wavelengths[0], process::iinput->wavelengths.size()*sizeof(int));
	cudaSubtractPoly(stoi(process::iinput->config_device["nCUDABLOCKS"]), stoi(process::iinput->config_device["nCUDATHREADSPERBLOCK"]),
		p_data_slices, p_data_spaxel_coeffs, poly_order + 1, wavelengths, process::d_datacube->slices.size(),
		process::d_datacube->slices[0]->getNumberOfElements());
	if (cudaThreadSynchronize() != cudaSuccess) {
		throw_error(CUDA_FAIL_SYNCHRONIZE);
	}

	culaShutdown();

	dmemory<Complex>::free(this_d_A);
	dmemory<Complex>::free(p_spaxel);
	dmemory<Complex*>::free(p_data_slices);
	dmemory<Complex>::free(p_data_spaxels);
	hmemory<Complex>::free(A);
	dmemory<Complex>::free(d_A);
	dmemory<Complex>::free(p_data_spaxel_coeffs);

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

std::vector<rectangle> process::growDatacubeToLargestDimensionSliceOnDevice() {
	std::vector<rectangle> last_regions;
	for (std::vector<dspslice*>::iterator it = process::d_datacube->slices.begin(); it != process::d_datacube->slices.end(); ++it) {
		last_regions.push_back((*it)->region);
	}
	d_datacube->grow(d_datacube->getLargestSliceRegion());
	return last_regions;
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
		throw_error(CPROCESS_REVERT_CROP_REGIONS_NOT_SET);
	}
	process::d_datacube->grow(process::last_crop_regions);

	// empty last crop factors
	process::last_crop_regions.clear();
}

void process::revertLastGrow() {
	if (process::last_grow_regions.size() != process::d_datacube->slices.size()) {
		throw_error(CPROCESS_REVERT_GROW_REGIONS_NOT_SET);
	}
	process::d_datacube->crop(process::last_grow_regions);

	// empty last grow factors
	process::last_grow_regions.clear();
}

void process::revertLastRescale() {
	if (process::last_rescale_regions.size() != process::d_datacube->slices.size()) {
		throw_error(CPROCESS_REVERT_RESCALE_REGIONS_NOT_SET);
	}
	std::vector<double> inverse_scale_factors;
	for (int i = 0; i < process::d_datacube->slices.size(); i++) {
		inverse_scale_factors[i] = (double)d_datacube->slices[i]->region.x_size / (double)process::last_rescale_regions[i].x_size;
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

std::vector<rectangle> process::rescaleDatacubeToReferenceWavelengthOnDevice(int reference_wavelength) {
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
	return last_regions;
}

void process::setDataToAmplitude() {
	for (std::vector<dspslice*>::iterator it = process::d_datacube->slices.begin(); it != process::d_datacube->slices.end(); ++it) {
		if (cudaSetComplexRealAsAmplitude(stoi(process::iinput->config_device["nCUDABLOCKS"]),
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
		process::last_crop_regions = process::cropDatacubeToSmallestDimensionSliceOnDevice();
		break;
	case D_SPAXEL_FIT_POLY_AND_SUBTRACT:
		sprintf(process::message_buffer, "%d\tPROCESS (%d/%d)\tfitting polynomial to spaxels and subtracting on device", process::exp_idx, stage, nstages);
		to_stdout(process::message_buffer);
		process::fitPolyToSpaxelAndSubtractOnDevice(
			stoi(process::iinput->stage_parameters[D_SPAXEL_FIT_POLY_AND_SUBTRACT]["POLY_ORDER"]));
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
		process::last_grow_regions = process::growDatacubeToLargestDimensionSliceOnDevice();
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
		process::last_grow_regions = process::rescaleDatacubeToReferenceWavelengthOnDevice(
			stoi(process::iinput->stage_parameters[D_RESCALE_DATACUBE_TO_REFERENCE_WAVELENGTH]["WAVELENGTH"]));
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