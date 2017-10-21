#pragma once

#include <stdio.h>
#include "cinput.h"
#include "cclparser.h"
#include "regions.h"

const double RESCALE_WAVELENGTH = 2.45; // micron

enum process_stages {
	COPY_DEVICE_DATACUBE_TO_HOST,
	COPY_HOST_DATACUBE_TO_DEVICE,
	D_CROP_TO_SMALLEST_DIMENSION,
	D_FFT,
	D_FFTSHIFT,
	D_IFFT,
	D_IFFTSHIFT,
	D_IRESCALE,
	D_RESCALE,
	D_SET_DATA_TO_AMPLITUDE,
	H_CROP_TO_EVEN_SQUARE,
	MAKE_DATACUBE_ON_HOST,
};

class process {
public:
	process() {};
	process(std::list<process_stages>, input*, clparser*, int);
	~process();
	void run();
	void step();
	input* iinput;
	clparser* iclparser;
	hcube* h_datacube;
	dcube* d_datacube;
	int exp_idx;
	std::list<process_stages> stages;
	char message_buffer[255];
private:
	void copyDeviceDatacubeToHost();
	void copyHostDatacubeToDevice();
	void correctOffsetOnDevice();
	void cropToEvenSquareOnHost();
	void cropToSmallestDimensionOnDevice();
	void fftOnDevice();
	void fftshiftOnDevice();
	void iFftOnDevice();
	void iFftshiftOnDevice();
	void iRescaleByWavelengthOnDevice();
	void makeDatacubeOnHost();
	void normaliseOnDevice();
	void rescaleByWavelengthOnDevice();
	void setDataToAmplitude();
};