#pragma once

#include <stdio.h>
#include <list>

#include "regions.h"
#include "ccube.h"
#include "cprocess.h"

enum process_stages {
	COPY_DEVICE_DATACUBE_TO_HOST,
	COPY_HOST_DATACUBE_TO_DEVICE,
	D_CROP_TO_SMALLEST_DIMENSION,
	D_FFT,
	D_FFTSHIFT,
	D_IFFT,
	D_IFFTSHIFT,
	D_RESCALE_DATACUBE_TO_PRE_RESCALE_SIZE,
	D_RESCALE_DATACUBE_TO_REFERENCE_WAVELENGTH,
	D_SPAXEL_FIT_POLY_AND_SUBTRACT,
	D_SET_DATA_TO_AMPLITUDE,
	H_CROP_TO_EVEN_SQUARE,
	MAKE_DATACUBE_ON_HOST
};

class input;
class process {
public:
	process() {};
	process(input*, int);
	~process();
	void run();
	input* iinput;
	hcube* h_datacube;
	dcube* d_datacube;
	int exp_idx;
	char message_buffer[255];
	std::list<process_stages> stages;
	std::vector<rectangle> pre_rescale_regions;
private:
	void copyDeviceDatacubeToHost();
	void copyHostDatacubeToDevice();
	void correctOffsetOnDevice();
	void cropToEvenSquareOnHost();
	void cropToSmallestDimensionOnDevice();
	void fitPolyToSpaxelAndSubtractOnDevice(int);
	void fftOnDevice();
	void fftshiftOnDevice();
	void iFftOnDevice();
	void iFftshiftOnDevice();
	void makeDatacubeOnHost();
	void normaliseOnDevice();
	void rescaleDatacubeToPreRescaleSizeOnDevice();
	std::vector<rectangle> rescaleDatacubeToReferenceWavelengthOnDevice(int);
	void setDataToAmplitude();
	void step(int, int);
};