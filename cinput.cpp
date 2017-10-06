#include "cinput.h"

#include <stdlib.h> 
#include <stdio.h>
#include <string>
#include <list>

#include <CCfits>
#include "cufft.h"
#include "rapidxml.hpp"

#include "ccube.h"

using namespace rapidxml;
using namespace CCfits;
using std::valarray;

input::input(std::string in_fits_filename, std::string in_params_filename, bool verbose) {
	/*
	This class houses all the necessary functions required to convert input FITS and parameter files into a 
	host cube instance.
	*/
	input::in_fits_filename = in_fits_filename;
	input::in_params_filename = in_params_filename;
	if (input::processParametersFile(in_params_filename, true)) {
		input::processFITSFile(in_fits_filename, true);
	}
}

hcube* input::makeCube(long n_exposure, bool verbose) {
	/* 
	This function constructs a host cube instance using the necessary class variables for the [n_exposure] exposure.
	*/
	if (input::state == CINPUT_OK) {
		try {

			std::size_t start = n_exposure*(input::dim[0]*input::dim[1]);
			std::size_t lengths[] = { input::dim[3], input::dim[0]*input::dim[1] };
			std::size_t strides[] = { input::dim[0]*input::dim[1]*input::dim[2], 1 };
			std::gslice exp_slice(start, std::valarray<std::size_t>(lengths, 2), std::valarray<std::size_t>(strides, 2));

			std::valarray<double> this_exposure_contents = input::data[exp_slice];

			hcube* datacube = new hcube(this_exposure_contents, std::vector<long>({ input::dim[0], input::dim[1], input::dim[3] }), input::wavelengths);

			return datacube;
		}
		catch (FitsException&) {
			input::state = CINPUT_FAULT_CONSTRUCT_CUBE;
			return NULL;
		}
	}
}

bool input::processFITSFile(string filename, bool verbose) {
	/*
    This function takes a FITS file and processes it, reading the FITS file data and dimensions into the class variables 
	[data] and [dim]. On success, it will set [state] to CINPUT_OK.
	*/
	input::readFITSFile(input::data, input::dim, filename, verbose);
	if (verbose) {
		printf("FITS file:\t\t\t%s\n", input::in_fits_filename.c_str());
		printf("Dimension 1:\t\t\t%d\n", input::dim[0]);
		printf("Dimension 2:\t\t\t%d\n", input::dim[1]);
		printf("Number of exposures:\t\t%d\n", input::dim[2]);
		printf("Number of spectral slices:\t%d\n", input::dim[3]);
	}

	input::state = CINPUT_OK;
	return 0;
}

bool input::processParametersFile(string filename, bool verbose) {
	/*
    This function takes an XML parameters file and processes it, creating a list of wavelengths for each slice of 
	the cube and populating the class variable [wavelengths].  On success, it will set [state] to 
	CINPUT_NOT_PROCESSED_FITS_FILE.
	*/
	input::readXMLFile(input::params, filename, verbose); // parse parameters into [params]

	double wmin, wmax;
	long wnum;
	xml_node<> *node = input::params.first_node()->first_node("parameters");
	for (xml_node<> *node_params = node->first_node(); node_params; node_params = node_params->next_sibling()) {
		for (xml_attribute<> *attr = node_params->first_attribute(); attr; attr = attr->next_attribute()) {
			if (strcmp(attr->name(), "name") == 0 && strcmp(attr->value(), "wmin") == 0) {
				wmin = atof(node_params->first_attribute("value")->value());
			}
			else if (strcmp(attr->name(), "name") == 0 && strcmp(attr->value(), "wmax") == 0) {
				wmax = atof(node_params->first_attribute("value")->value());
			}
			else if (strcmp(attr->name(), "name") == 0 && strcmp(attr->value(), "wnum") == 0) {
				wnum = atoi(node_params->first_attribute("value")->value());
			}
		}
	}

	float wavelength_increment = (wmax - wmin) / ((double)(wnum-1));
	for (float w = wmin; w <= wmax + (wavelength_increment/2.); w += wavelength_increment) {		// need to make sure we get last wavelength
		input::wavelengths.push_back(w);
	}

	if (verbose) {
		printf("Parameters file:\t\t%s\n", input::in_params_filename.c_str());
		printf("Wavelength start (micron):\t%.2f\n", wmin);
		printf("Wavelength end (micron):\t%.2f\n", wmax);
		printf("Wavelength nbins:\t\t%d\n", wnum);
	}
	input::state = CINPUT_NOT_PROCESSED_FITS_FILE;
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
		input::state = CINPUT_FAULT_READ_FITS_ERROR;
		return false;
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

