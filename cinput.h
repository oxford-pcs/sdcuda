#pragma once

#include <string>

#include <CCfits>
#include "rapidxml.hpp"

#include "ccube.h"

using namespace rapidxml;

class input {
public:
	input(std::string, std::string, bool);
	~input() {};
	hcube* makeCube(long, bool);
	std::vector<long> dim;
	std::valarray<double> data;
	std::string in_fits_filename, in_params_filename;
	xml_document<> config;
	xml_document<> params;
	std::vector<double> wavelengths;
private:
	bool readXMLFile(xml_document<>&, string, bool);
	bool readFITSFile(std::valarray<double>&, std::vector<long>&, string, bool);
	bool processConfigFile(string, bool);
	bool processSimulationParametersFile(string, bool);
	bool processFITSFile(string, bool);
};
