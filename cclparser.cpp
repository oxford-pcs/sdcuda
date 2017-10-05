#include "cclparser.h"

#include <stdio.h>
#include <ctype.h>
#include <string>
#include <iostream>
#include "getopt.h"

clparser::clparser(int argc, char **argv) {
	/*
	This class parses command line input and checks if it's OK.
	*/
	clparser::argc = argc;
	clparser::argv = argv;
	if (clparser::parse()) {
		clparser::check();
	}
}

bool clparser::check() {
	/* 
    This function checks if we have both the correct number of inputs and 
    if the inputs are valid, e.g. if input files exist. On success, it will
	set [state] to CCLPARSER_OK.
	*/
	if (clparser::nargs != 3) {
		fprintf(stderr, "Invalid number of parameters specified.");
		clparser::state = CCLPARSER_FAULT_INVALID_NUMBER_OF_PARAMETERS;
		return false;
	} else if (!is_file_existing(clparser::in_FITS_filename)) {
		fprintf(stderr, "Input FITS file does not exist.");
		clparser::state = CCLPASRER_FAULT_INPUT_FITS_FILE_NO_EXIST;
		return false;
	} else if (!is_file_existing(clparser::in_params_filename)) {
		fprintf(stderr, "Input parameters file does not exist.");
		clparser::state = CCLPARSER_FAULT_INPUT_PARAMETERS_FILE_NO_EXIST;
		return false;
	} 
	clparser::state = CCLPARSER_OK;
	return true;
}

bool clparser::parse() {
	/*
	This function parses the command line input arguments. On success, it will 
	set [state] to CCLPARSER_WARN_NOT_CHECKED.
	*/
	int opterr = 0;
	int c;
	while ((c = getopt(clparser::argc, clparser::argv, "i:p:o:")) != -1)
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
				clparser::state = CCLPARSER_FAULT_MISSING_ARGUMENT;
				fprintf(stderr, "Option -%c requires an argument.\n", optopt);
				return false;
			} else if (isprint(optopt)) {
				fprintf(stderr, "Unknown option `-%c'.\n", optopt);
				clparser::state = CCLPARSER_FAULT_UNKNOWN_OPTION;
				return false;
			} else {
				fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
				clparser::state = CCLPARSER_FAULT_UNKNOWN_CHARACTER;
				return false;
			}
		default:
			clparser::state = CCLPARSER_FAULT_UNKNOWN_CONDITION;
			return false;
	}
	clparser::state = CCLPARSER_NOT_CHECKED;
	return true;
}