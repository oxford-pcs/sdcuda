#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <map>

#include "logger.h"

enum errors {
	CCLPARSER_MISSING_ARGUMENT = -1,
	CCLPARSER_UNKNOWN_OPTION = -2,
	CCLPARSER_UNKNOWN_CHARACTER = -3,
	CCLPARSER_INVALID_NUMBER_OF_PARAMETERS = -4,
	CCLPARSER_INPUT_FITS_FILE_NO_EXIST = -5,
	CCLPARSER_INPUT_PARAMETERS_FILE_NO_EXIST = -6,
	CCLPARSER_OPTION_NO_ARGUMENT = -7,
	CCLPARSER_UNKNOWN_CONDITION = -8,
	CCUBE_BAD_DOMAIN = -10,
	CCUBE_FAIL_WRITE = -11,
	CCUBE_FAIL_INTEGRITY_CHECK = -12,
	CINPUT_READ_FITS_ERROR = -20,
	CINPUT_FAIL_CONSTRUCT_CUBE = -21,
	CPROCESS_UNKNOWN_STAGE = -30,
	CUDA_FAIL_SYNCHRONIZE = -100
};

const std::map<errors, char*> error_messages = {
	{ CCLPARSER_MISSING_ARGUMENT, "CCLPARSER_MISSING_ARGUMENT" },
	{ CCLPARSER_UNKNOWN_OPTION, "CCLPARSER_UNKNOWN_OPTION" },
	{ CCLPARSER_UNKNOWN_CHARACTER, "CCLPARSER_UNKNOWN_CHARACTER" },
	{ CCLPARSER_INVALID_NUMBER_OF_PARAMETERS, "CCLPARSER_INVALID_NUMBER_OF_PARAMETERS" },
	{ CCLPARSER_INPUT_FITS_FILE_NO_EXIST, "CCLPARSER_INPUT_FITS_FILE_NO_EXIST" },
	{ CCLPARSER_INPUT_PARAMETERS_FILE_NO_EXIST, "CCLPARSER_INPUT_PARAMETERS_FILE_NO_EXIST" },
	{ CCLPARSER_UNKNOWN_CONDITION, "CCLPARSER_UNKNOWN_CONDITION" },
	{ CCUBE_BAD_DOMAIN, "CCUBE_BAD_DOMAIN" },
	{ CCUBE_FAIL_WRITE, "CCUBE_FAIL_WRITE" },
	{ CCUBE_FAIL_INTEGRITY_CHECK, "CCUBE_FAIL_INTEGRITY_CHECK" },
	{ CINPUT_READ_FITS_ERROR, "CINPUT_READ_FITS_ERROR" },
	{ CINPUT_FAIL_CONSTRUCT_CUBE, "CINPUT_FAIL_CONSTRUCT_CUBE" },
	{ CPROCESS_UNKNOWN_STAGE, "CPROCESS_UNKNOWN_STAGE" },
	{ CUDA_FAIL_SYNCHRONIZE, "CUDA_FAIL_SYNCHRONIZE" }
};

void throw_error(errors);