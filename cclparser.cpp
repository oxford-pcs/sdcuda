#include "cclparser.h"

#include <stdio.h>
#include <ctype.h>
#include <string>
#include <iostream>
#include "getopt.h"

cclparser::cclparser(int argc, char **argv) {
	/*
	This class parses command line input and checks if it's OK.
	*/
	cclparser::argc = argc;
	cclparser::argv = argv;
	if (cclparser::parse()) {
		cclparser::check();
	}
}

bool cclparser::check() {
	/* 
    This function checks if we have both the correct number of inputs and 
    if the inputs are valid, e.g. if input files exist. On success, it will
	set [state] to CCLPARSER_OK.
	*/
	if (cclparser::nargs != 3) {
		fprintf(stderr, "Invalid number of parameters specified.");
		cclparser::state = CCLPARSER_FAULT_INVALID_NUMBER_OF_PARAMETERS;
		return false;
	} else if (!is_file_existing(cclparser::in_FITS_filename)) {
		fprintf(stderr, "Input FITS file does not exist.");
		cclparser::state = CCLPASRER_FAULT_INPUT_FITS_FILE_NO_EXIST;
		return false;
	} else if (!is_file_existing(cclparser::in_params_filename)) {
		fprintf(stderr, "Input parameters file does not exist.");
		cclparser::state = CCLPARSER_FAULT_INPUT_PARAMETERS_FILE_NO_EXIST;
		return false;
	} 
	cclparser::state = CCLPARSER_OK;
	return true;
}

bool cclparser::parse() {
	/*
	This function parses the command line input arguments. On success, it will 
	set [state] to CCLPARSER_WARN_NOT_CHECKED.
	*/
	int opterr = 0;
	int c;
	while ((c = getopt(cclparser::argc, cclparser::argv, "i:p:o:")) != -1)
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
		case 'o':
			out_FITS_filename = std::string(optarg);
			nargs++;
			break;
		case '?':
			if (optopt == 'i' || optopt == 'p' || optopt == 'o') {
				cclparser::state = CCLPARSER_FAULT_MISSING_ARGUMENT;
				fprintf(stderr, "Option -%c requires an argument.\n", optopt);
				return false;
			} else if (isprint(optopt)) {
				fprintf(stderr, "Unknown option `-%c'.\n", optopt);
				cclparser::state = CCLPARSER_FAULT_UNKNOWN_OPTION;
				return false;
			} else {
				fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
				cclparser::state = CCLPARSER_FAULT_UNKNOWN_CHARACTER;
				return false;
			}
		default:
			cclparser::state = CCLPARSER_FAULT_UNKNOWN_CONDITION;
			return false;
	}
	cclparser::state = CCLPARSER_NOT_CHECKED;
	return true;
}