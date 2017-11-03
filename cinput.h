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
	{ "D_CROP_DATACUBE_TO_SMALLEST_DIMENSION_SLICE", D_CROP_DATACUBE_TO_SMALLEST_DIMENSION_SLICE },
	{ "D_SPAXEL_FIT_POLY_AND_SUBTRACT", D_SPAXEL_FIT_POLY_AND_SUBTRACT },
	{ "D_FFT", D_FFT },
	{ "D_FFTSHIFT", D_FFTSHIFT },
	{ "D_GROW_DATACUBE_TO_LARGEST_DIMENSION_SLICE", D_GROW_DATACUBE_TO_LARGEST_DIMENSION_SLICE },
	{ "D_IFFT", D_IFFT },
	{ "D_IFFTSHIFT", D_IFFTSHIFT },
	{ "D_REVERT_LAST_CROP", D_REVERT_LAST_CROP },
	{ "D_REVERT_LAST_GROW", D_REVERT_LAST_GROW },
	{ "D_RESCALE_DATACUBE_TO_PRE_RESCALE_SIZE", D_REVERT_LAST_RESCALE },
	{ "D_RESCALE_DATACUBE_TO_REFERENCE_WAVELENGTH", D_RESCALE_DATACUBE_TO_REFERENCE_WAVELENGTH },
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
	std::map<std::string, std::string> config_host;
	std::map<std::string, std::string> config_device;
	std::list<process_stages> stages;
	std::map<process_stages, std::map<std::string, std::string>> stage_parameters;
private:
	void readXMLFile(xml_document<>&, string, bool);
	void readFITSFile(std::valarray<double>&, std::vector<long>&, string, bool);
	void processConfigFile(string, bool);
	void processSimulationParametersFile(string, bool);
	void processFITSFile(string, bool);
};
