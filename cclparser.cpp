#include "cclparser.h"

#include <stdio.h>
#include <ctype.h>
#include <string>
#include <iostream>
#include "getopt.h"
#include "errors.h"

clparser::clparser(int argc, char **argv) {
	/*
	This class parses command line input and checks if it's OK.
	*/
	clparser::argc = argc;
	clparser::argv = argv;
	clparser::parse();
	clparser::check();
}

void clparser::check() {
	/* 
    This function checks if we have both the correct number of inputs and if the inputs are valid, e.g. if input files exist.
	*/
	if (clparser::nargs != 4) {
		throw_error(CCLPARSER_INVALID_NUMBER_OF_PARAMETERS);
	} else if (!is_file_existing(clparser::in_FITS_filename)) {
		throw_error(CCLPARSER_INPUT_FITS_FILE_NO_EXIST);
	} else if (!is_file_existing(clparser::in_params_filename)) {
		throw_error(CCLPARSER_INPUT_PARAMETERS_FILE_NO_EXIST);
	} else if (!is_file_existing(clparser::in_config_filename)) {
		throw_error(CCLPARSER_INPUT_CONFIG_FILE_NO_EXIST);
	}
}

void clparser::parse() {
	/*
	This function parses the command line input arguments.
	*/
	int opterr = 0;
	int c;
	while ((c = getopt(clparser::argc, clparser::argv, "i:p:o:c:|")) != -1)
		switch (c)
		{
		case 'i':
			in_FITS_filename = std::string(optarg);
			nargs++;
			break;
		case 'p':
			in_params_filename = std::string(optarg);
			nargs++;
			break;
		case 'c':
			in_config_filename = std::string(optarg);
			nargs++;
			break;
		case 'o':
			out_FITS_filename = std::string(optarg);
			nargs++;
			break;
		case '?':
			if (optopt == 'i' || optopt == 'p' || optopt == 'o' || optopt == 'c') {
				throw_error(CCLPARSER_OPTION_NO_ARGUMENT);
			} else if (isprint(optopt)) {
				throw_error(CCLPARSER_UNKNOWN_OPTION);
			} else {
				throw_error(CCLPARSER_UNKNOWN_CHARACTER);
			}
		default:
			throw_error(CCLPARSER_UNKNOWN_CONDITION);
		}
}