#include "cinput.h"

#include <stdlib.h> 
#include <stdio.h>
#include <string>
#include <list>
#include <map>

#include <CCfits>
#include "rapidxml.hpp"

#include "logger.h"
#include "ccube.h"
#include "cprocess.h"

using namespace rapidxml;
using namespace CCfits;
using std::valarray;

input::input(std::string in_fits_filename, std::string in_params_filename, std::string in_config_filename, bool verbose) {
	/*
	This class reads FITS data, config items and simulation parameters from files specfied at the command line. Input 
	instances are used to construct a process instance.
	*/
	input::in_fits_filename = in_fits_filename;
	input::in_params_filename = in_params_filename;
	input::in_config_filename = in_config_filename;

	if (verbose) {
		printf("Config file:\t\t\t%s\n", input::in_config_filename.c_str());
		printf("FITS file:\t\t\t%s\n", input::in_fits_filename.c_str());
		printf("Parameters file:\t\t%s\n", input::in_params_filename.c_str());

		size_t free_byte;
		size_t total_byte;
		int cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
		if (cuda_status != cudaSuccess){
			throw_error(CUDA_FAIL_GET_DEVICE_MEMORY);
		}
		double free_db = (double)free_byte/1000000.;
		double total_db = (double)total_byte/1000000.;
		printf("Free device memory:\t\t%.2f Mb \n", free_db);
		printf("Total device memory:\t\t%.2f Mb\n", total_db);
	}
	input::processConfigFile(in_config_filename, true);
	input::processFITSFile(in_fits_filename, true);
	input::processSimulationParametersFile(in_params_filename, true);
}


hcube* input::makeHostCube(long n_exposure, bool verbose) {
	/* 
	Construct a host cube instance from the exposure with index [n_exposure].
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

void input::processConfigFile(string filename, bool verbose) {
	/*
	Process a config file.
	*/
	input::readXMLFile(input::config, filename, verbose);

	xml_node<> *node;
	std::string stage_name, stage_value;
	std::string param_name, param_value;

	// populate host dictionary
	node = input::config.first_node()->first_node("host");
	for (xml_node<> *node_host = node->first_node(); node_host; node_host = node_host->next_sibling()) {
		for (xml_node<> *node_param = node_host->first_node(); node_param; node_param = node_param->next_sibling()) {
			if (strcmp(node_param->name(), "name") == 0) {
				param_name = std::string(node_param->value());
			} else if (strcmp(node_param->name(), "value") == 0) {
				param_value = std::string(node_param->value());
			}
		}
		input::config_host[param_name] = param_value;
	}

	// populate device dictionary
	node = input::config.first_node()->first_node("device");
	for (xml_node<> *node_device = node->first_node(); node_device; node_device = node_device->next_sibling()) {
		for (xml_node<> *node_param = node_device->first_node(); node_param; node_param = node_param->next_sibling()) {
			if (strcmp(node_param->name(), "name") == 0) {
				param_name = std::string(node_param->value());
			} else if (strcmp(node_param->name(), "value") == 0) {
				param_value = std::string(node_param->value());
			}
		}
		input::config_device[param_name] = param_value;
	}

	// populate process stage dictionary
	node = input::config.first_node()->first_node("process");
	for (xml_node<> *node_process = node->first_node(); node_process; node_process = node_process->next_sibling()) {
		std::map<std::string, std::string> this_stage_params;
		for (xml_node<> *node_stage = node_process->first_node(); node_stage; node_stage = node_stage->next_sibling()) {
			if (strcmp(node_stage->name(), "name") == 0) {
				stage_name = std::string(node_stage->value());
			} else if (strcmp(node_stage->name(), "param") == 0) {
				for (xml_node<> *node_param = node_stage->first_node(); node_param; node_param = node_param->next_sibling()) {
					if (strcmp(node_param->name(), "name") == 0) {
						param_name = std::string(node_param->value());
					} else if (strcmp(node_param->name(), "value") == 0) {
						param_value = std::string(node_param->value());
					}
				}
				this_stage_params[param_name] = param_value;
			}
		}
		try {
			input::stages.push_back(process_stages_mapping.at(stage_name));
			input::stage_parameters[process_stages_mapping.at(stage_name)] = this_stage_params;
		}
		catch (const std::exception& e) {
			if (strcmp("invalid map<K, T> key", (&e)->what()) == 0) {
				throw_error(CINPUT_UNRECOGNISED_STAGE);
			}
		}
	}

	if (verbose) {
		printf("Max number of processes:\t%d\n", atoi(input::config_host["nCPUCORES"].c_str()));
		printf("Number of CUDA blocks:\t\t%d\n", atoi(input::config_device["nCUDABLOCKS"].c_str()));
		printf("Threads per CUDA block:\t\t%d\n", atoi(input::config_device["nCUDATHREADSPERBLOCK"].c_str()));
	}
}

void input::processFITSFile(string filename, bool verbose) {
	/*
    Take a FITS file and process it, reading the FITS file data and dimensions into the class variables [data] and [dim].
	*/
	input::readFITSFile(input::data, input::dim, filename, verbose);
	if (verbose) {
		printf("FITS Dimension 1:\t\t%d\n", input::dim[0]);
		printf("FITS Dimension 2:\t\t%d\n", input::dim[1]);
		printf("Number of exposures:\t\t%d\n", input::dim[2]);
		printf("Number of spectral slices:\t%d\n", input::dim[3]);
	}
}

void input::processSimulationParametersFile(string filename, bool verbose) {
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
		printf("Wavelength start (nm):\t\t%d\n", wmin);
		printf("Wavelength end (nm):\t\t%d\n", wmax);
		printf("Wavelength nbins:\t\t%d\n", wnum);
	}
}


void input::readFITSFile(std::valarray<double> &data, std::vector<long> &dim, string filename, bool verbose) {
	/*
	Read the data from a FITS file with a given file path [filename] into a valarray [data], 
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
}

void input::readXMLFile(xml_document<> &doc, string filename, bool verbose) {
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
}

