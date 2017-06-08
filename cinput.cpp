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
	input::in_fits_filename = in_fits_filename;
	input::in_params_filename = in_params_filename;
	input::processParametersFile(in_params_filename, true);
	input::processFITSFile(in_fits_filename, true);
}

hcube* input::makeCube(long n_exposure, bool verbose) {
	try {
		if (verbose) {
			printf("\n");
			printf("Making cube for exposure number: %d\n", n_exposure);
		}

		std::size_t start = n_exposure*(input::dim[0]*input::dim[1]);
		std::size_t lengths[] = { input::dim[3], input::dim[0]*input::dim[1] };
		std::size_t strides[] = { input::dim[0]*input::dim[1]*input::dim[2], 1 };
		std::gslice exp_slice(start, std::valarray<std::size_t>(lengths, 2), std::valarray<std::size_t>(strides, 2));

		std::valarray<double> this_exposure_contents = input::data[exp_slice];

		hcube* datacube = new hcube(this_exposure_contents, std::vector<long>({ input::dim[0], input::dim[1], input::dim[3] }), input::wavelengths);

		return datacube;
	}
	catch (FitsException&) {
		return NULL;
	}
}

int input::processFITSFile(string filename, bool verbose) {
	input::readFITSFile(input::data, input::dim, filename, verbose);

	if (verbose) {
		printf("\n");
		printf("FITS File Information\n");
		printf("----------------\n\n");
		printf("Dimension 1:\t\t\t%d\n", input::dim[0]);
		printf("Dimension 2:\t\t\t%d\n", input::dim[1]);
		printf("Number of exposures:\t\t%d\n", input::dim[2]);
		printf("Number of spectral slices:\t%d\n", input::dim[3]);
	}

	return 0;
}

int input::processParametersFile(string filename, bool verbose) {
	input::readXMLFile(input::params, filename, verbose);

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

	float wavelength_increment = (wmax - wmin) / (double)wnum;
	for (float w = wmin; w < wmax; w += wavelength_increment) {
		input::wavelengths.push_back(w);
	}

	if (verbose) {
		printf("\n");
		printf("FITS file:\t\t%s\n", input::in_fits_filename.c_str());
		printf("Parameters file:\t%s\n\n", input::in_params_filename.c_str());
		printf("Wavelength start (micron):\t%.2f\n", wmin);
		printf("Wavelength end (micron):\t%.2f\n", wmax);
		printf("Wavelength nbins:\t\t%d\n", wnum);
	}
	return 0;
}

int input::readFITSFile(std::valarray<double> &data, std::vector<long> &dim, string filename, bool verbose) {
	std::auto_ptr<FITS> pInfile;
	try {
		pInfile.reset(new FITS(filename, Read, true));
	}
	catch (FitsException&) {
		return 1;
	}
	PHDU &image = pInfile->pHDU();
	dim.push_back(image.axis(0));
	dim.push_back(image.axis(1));
	dim.push_back(image.axis(2));	// exposures
	dim.push_back(image.axis(3));	// spectral slices

	image.read(data, 1, std::accumulate(begin(dim), end(dim), 1, std::multiplies<long>()));

	return 0;
}

int input::readXMLFile(xml_document<> &doc, string filename, bool verbose) {
	FILE *f = fopen(filename.c_str(), "rb");
	fseek(f, 0, SEEK_END);
	long fsize = ftell(f);
	fseek(f, 0, SEEK_SET);

	char *contents = (char *)malloc(fsize + 1);
	fread(contents, fsize, 1, f);
	fclose(f);

	contents[fsize] = 0;

	doc.parse<0>(contents);
	return 0;
}

