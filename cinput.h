#pragma once

#include <string>

#include <CCfits>
#include "rapidxml.hpp"

#include "ccube.h"

using namespace rapidxml;

enum cinput_states {
	CINPUT_NOT_PROCESSED_FITS_FILE = 2,
	CINPUT_NOT_PROCESSED_PARAMETERS_FILE = 1,
	CINPUT_OK = 0,
	CINPUT_FAULT_READ_FITS_ERROR = -1,
	CINPUT_FAULT_CONSTRUCT_CUBE = -2
};

class input {
public:
	input(std::string, std::string, bool);
	~input() {};
	hcube* makeCube(long, bool);
	bool readXMLFile(xml_document<>&, string, bool);
	bool readFITSFile(std::valarray<double>&, std::vector<long>&, string, bool);
	std::vector<long> dim;
	std::valarray<double> data;
	std::string in_fits_filename, in_params_filename;
	xml_document<> params;
	cinput_states state = CINPUT_NOT_PROCESSED_PARAMETERS_FILE;
	std::vector<double> wavelengths;
private:
	bool processParametersFile(string, bool);
	bool processFITSFile(string, bool);
};
