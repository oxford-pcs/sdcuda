#pragma once

#include <string>

#include <CCfits>
#include "rapidxml.hpp"

#include "ccube.h"
#include "cprocess.h"

using namespace rapidxml;

const std::map<std::string, process_stages> process_stages_mapping {
	{ "COPY_DEVICE_DATACUBE_TO_HOST", COPY_DEVICE_DATACUBE_TO_HOST },
	{ "COPY_HOST_DATACUBE_TO_DEVICE", COPY_HOST_DATACUBE_TO_DEVICE },
	{ "D_CROP_TO_SMALLEST_DIMENSION", D_CROP_TO_SMALLEST_DIMENSION },
	{ "D_FFT", D_FFT },
	{ "D_FFTSHIFT", D_FFTSHIFT },
	{ "D_IFFT", D_IFFT },
	{ "D_IFFTSHIFT", D_IFFTSHIFT },
	{ "D_IRESCALE", D_IRESCALE },
	{ "D_RESCALE", D_RESCALE },
	{ "D_SET_DATA_TO_AMPLITUDE", D_SET_DATA_TO_AMPLITUDE },
	{ "H_CROP_TO_EVEN_SQUARE", H_CROP_TO_EVEN_SQUARE },
	{ "MAKE_DATACUBE_ON_HOST", MAKE_DATACUBE_ON_HOST }
};

class input {
public:
	input(std::string, std::string, std::string, bool);
	~input() {};
	hcube* makeCube(long, bool);
	std::vector<long> dim;
	std::valarray<double> data;
	std::string in_fits_filename, in_params_filename, in_config_filename;
	xml_document<> config;
	xml_document<> params;
	std::vector<int> wavelengths;
	int nCPUCORES;
	int nCUDABLOCKS;
	int nCUDATHREADSPERBLOCK;
	int RESCALE_WAVELENGTH;					// nm
	std::list<process_stages> stages;
private:
	void readXMLFile(xml_document<>&, string, bool);
	void readFITSFile(std::valarray<double>&, std::vector<long>&, string, bool);
	void processConfigFile(string, bool);
	void processSimulationParametersFile(string, bool);
	void processFITSFile(string, bool);
};
