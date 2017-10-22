#include "cinput.h"

#include <stdlib.h> 
#include <stdio.h>
#include <string>
#include <list>

#include <CCfits>
#include "cufft.h"
#include "rapidxml.hpp"

#include "logger.h"
#include "ccube.h"
#include "cprocess.h"

using namespace rapidxml;
using namespace CCfits;
using std::valarray;

input::input(std::string in_fits_filename, std::string in_params_filename, std::string in_config_filename, bool verbose) {
	/*
	This class houses all the necessary functions required to convert input FITS and parameter files into a 
	host cube instance.
	*/
	input::in_fits_filename = in_fits_filename;
	input::in_params_filename = in_params_filename;
	input::in_config_filename = in_config_filename;

	input::processSimulationParametersFile(in_params_filename, true);
	input::processConfigFile(in_config_filename, true);
	input::processFITSFile(in_fits_filename, true);
}

hcube* input::makeCube(long n_exposure, bool verbose) {
	/* 
	This function constructs a host cube instance using the necessary class variables for the [n_exposure] exposure.
	*/
	try {
		std::size_t start = n_exposure*(input::dim[0]*input::dim[1]);
		std::size_t lengths[] = { input::dim[3], input::dim[0]*input::dim[1] };
		std::size_t strides[] = { input::dim[0]*input::dim[1]*input::dim[2], 1 };
		std::gslice exp_slice(start, std::valarray<std::size_t>(lengths, 2), std::valarray<std::size_t>(strides, 2));

		std::valarray<double> this_exposure_contents = input::data[exp_slice];

		hcube* datacube = new hcube(this_exposure_contents, std::vector<long>({ input::dim[0], input::dim[1], input::dim[3] }), input::wavelengths);

		return datacube;
	} catch (FitsException&) {
		throw_error(CINPUT_FAIL_CONSTRUCT_CUBE);
	}
}

bool input::processConfigFile(string filename, bool verbose) {
	input::readXMLFile(input::config, filename, verbose); // parse parameters into [config]
	xml_node<> *node;
	//host
	node = input::config.first_node()->first_node("host");
	for (xml_node<> *node_host = node->first_node(); node_host; node_host = node_host->next_sibling()) {
		for (xml_attribute<> *attr = node_host->first_attribute(); attr; attr = attr->next_attribute()) {
			if (strcmp(attr->name(), "name") == 0 && strcmp(attr->value(), "nCPUCORES") == 0) {
				input::nCPUCORES = atoi(node_host->first_attribute("value")->value());
			}
		}
	}
	// device
	node = input::config.first_node()->first_node("device");
	for (xml_node<> *node_device = node->first_node(); node_device; node_device = node_device->next_sibling()) {
		for (xml_attribute<> *attr = node_device->first_attribute(); attr; attr = attr->next_attribute()) {
			if (strcmp(attr->name(), "name") == 0 && strcmp(attr->value(), "nCUDABLOCKS") == 0) {
				input::nCUDABLOCKS = atoi(node_device->first_attribute("value")->value());
			} else if (strcmp(attr->name(), "name") == 0 && strcmp(attr->value(), "nCUDATHREADSPERBLOCK") == 0) {
				input::nCUDATHREADSPERBLOCK = atoi(node_device->first_attribute("value")->value());
			}		
		}
	}
	// process
	node = input::config.first_node()->first_node("process");
	for (xml_node<> *node_process = node->first_node(); node_process; node_process = node_process->next_sibling()) {
		for (xml_attribute<> *attr = node_process->first_attribute(); attr; attr = attr->next_attribute()) {
			if (strcmp(attr->name(), "name") == 0 && strcmp(attr->value(), "RESCALE_WAVELENGTH") == 0) {
				input::RESCALE_WAVELENGTH = atoi(node_process->first_attribute("value")->value());
			} else if (strcmp(attr->name(), "name") == 0 && strcmp(attr->value(), "STAGES") == 0) {
				for (xml_node<> *node_stages = node_process->first_node(); node_stages; node_stages = node_stages->next_sibling()) {
					for (xml_attribute<> *attr = node_stages->first_attribute(); attr; attr = attr->next_attribute()) {
						if (strcmp(attr->name(), "name") == 0 && strcmp(attr->value(), "stage") == 0) {
							std::string stage_str = std::string(node_stages->first_attribute("value")->value());
							input::stages.push_back(process_stages_mapping.at(stage_str));
						}
					}
				}
			}
		}
	}
	return true;
}

bool input::processFITSFile(string filename, bool verbose) {
	/*
    This function takes a FITS file and processes it, reading the FITS file data and dimensions into the class variables 
	[data] and [dim].
	*/
	input::readFITSFile(input::data, input::dim, filename, verbose);
	if (verbose) {
		printf("FITS file:\t\t\t%s\n", input::in_fits_filename.c_str());
		printf("Dimension 1:\t\t\t%d\n", input::dim[0]);
		printf("Dimension 2:\t\t\t%d\n", input::dim[1]);
		printf("Number of exposures:\t\t%d\n", input::dim[2]);
		printf("Number of spectral slices:\t%d\n", input::dim[3]);
	}
	return true;
}

bool input::processSimulationParametersFile(string filename, bool verbose) {
	/*
    This function takes an XML parameters file and processes it, creating a list of wavelengths for each slice of 
	the cube and populating the class variable [wavelengths]. 
	*/
	input::readXMLFile(input::params, filename, verbose); // parse parameters into [params]

	int wmin, wmax;
	long wnum;
	xml_node<> *node = input::params.first_node()->first_node("parameters");
	for (xml_node<> *node_params = node->first_node(); node_params; node_params = node_params->next_sibling()) {
		for (xml_attribute<> *attr = node_params->first_attribute(); attr; attr = attr->next_attribute()) {
			if (strcmp(attr->name(), "name") == 0 && strcmp(attr->value(), "wmin") == 0) {
				wmin = round(atof(node_params->first_attribute("value")->value())*1000);		// input is in micron
			}
			else if (strcmp(attr->name(), "name") == 0 && strcmp(attr->value(), "wmax") == 0) {
				wmax = round(atof(node_params->first_attribute("value")->value())*1000);		// input is in micron
			}
			else if (strcmp(attr->name(), "name") == 0 && strcmp(attr->value(), "wnum") == 0) {
				wnum = atoi(node_params->first_attribute("value")->value());
			}
		}
	}
	float wavelength_increment = (wmax - wmin) / (wnum-1);
	for (int w = wmin; w <= wmax;  w += wavelength_increment) {
		input::wavelengths.push_back(w);
	}
	if (verbose) {
		printf("Parameters file:\t\t%s\n", input::in_params_filename.c_str());
		printf("Wavelength start (micron):\t%.2f\n", wmin);
		printf("Wavelength end (micron):\t%.2f\n", wmax);
		printf("Wavelength nbins:\t\t%d\n", wnum);
	}
	return true;
}

bool input::readFITSFile(std::valarray<double> &data, std::vector<long> &dim, string filename, bool verbose) {
	/*
	This function reads the data from a FITS file with a given file path [filename] into a valarray [data], 
	populating [dim] with the dimensions of the image.
	*/
	std::auto_ptr<FITS> pInfile;
	try {
		pInfile.reset(new FITS(filename, Read, true));
	}
	catch (FitsException&) {
		throw_error(CINPUT_READ_FITS_ERROR);
	}

	PHDU &image = pInfile->pHDU();
	input::dim.push_back(image.axis(0));
	input::dim.push_back(image.axis(1));
	input::dim.push_back(image.axis(2));	// exposures
	input::dim.push_back(image.axis(3));	// spectral slices

    image.read(data, 1, std::accumulate(begin(dim), end(dim), 1, std::multiplies<long>()));

	return true;
}

bool input::readXMLFile(xml_document<> &doc, string filename, bool verbose) {
	/*
	This function reads an XML file with a given file path [filename], parsing it into rapidxml 
	xml_document [doc].
	*/
	FILE *f = fopen(filename.c_str(), "rb");
	fseek(f, 0, SEEK_END);
	long fsize = ftell(f);
	fseek(f, 0, SEEK_SET);

	char *contents = (char *)malloc(fsize + 1);
	fread(contents, fsize, 1, f);
	fclose(f);

	contents[fsize] = 0;

	doc.parse<0>(contents);
	return true;
}

