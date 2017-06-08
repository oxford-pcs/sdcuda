#pragma once

#include <string>

#include <CCfits>

#include "ccube.h"
#include "rapidxml.hpp"

using namespace rapidxml;

class input {
public:
	input(std::string, std::string, bool);
	~input() {};
	hcube* makeCube(long, bool);
	int readXMLFile(xml_document<>&, string, bool);
	int readFITSFile(std::valarray<double>&, std::vector<long>&, string, bool);
	std::vector<long> dim;
	std::valarray<double> data;
	std::string in_fits_filename, in_params_filename;
	xml_document<> params;
	std::vector<double> wavelengths;
private:
	int processParametersFile(string, bool);
	int processFITSFile(string, bool);
};
