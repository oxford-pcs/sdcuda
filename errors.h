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
	CCLPARSER_INPUT_CONFIG_FILE_NO_EXIST = -7,
	CCLPARSER_OPTION_NO_ARGUMENT = -8,
	CCLPARSER_UNKNOWN_CONDITION = -9,
	CCUBE_BAD_DOMAIN = -10,
	CCUBE_FAIL_WRITE = -11,
	CCUBE_FAIL_INTEGRITY_CHECK = -12,
	CINPUT_READ_FITS_ERROR = -20,
	CINPUT_FAIL_CONSTRUCT_CUBE = -21,
	CINPUT_UNRECOGNISED_STAGE = -22,
	CPROCESS_UNKNOWN_STAGE = -30,
	CPROCESS_REVERT_CROP_REGIONS_INVALID = -31,
	CPROCESS_REVERT_GROW_REGIONS_INVALID = -32,
	CPROCESS_REVERT_RESCALE_REGIONS_INVALID = -33,
	CPROCESS_FIT_AND_SUBTRACT_POLY_FAIL_INTEGRITY_CHECK = -34,
	CUDA_FAIL_SYNCHRONIZE = -100,
	CUDA_FAIL_MEMCPY_HH = -101,
	CUDA_FAIL_MEMCPY_HD = -102,
	CUDA_FAIL_MEMCPY_DH = -103,
	CUDA_FAIL_MEMCPY_DD = -104,
	CUDA_FAIL_FREE_MEMORY_H = -105,
	CUDA_FAIL_FREE_MEMORY_D = -106,
	CUDA_FAIL_ALLOCATE_MEMORY_H = -107,
	CUDA_FAIL_ALLOCATE_MEMORY_D = -108,
	CUDA_FAIL_SET_MEMORY_D = -109,
	CUDA_FFT_FAIL_CREATE_PLAN = -110,
	CUDA_FFT_FAIL_EXECUTE_PLAN = -111,
	CUDA_FAIL_GET_DEVICE_MEMORY = -112,
	CUDA_FAIL_GET_SPAXEL_DATA_2D = -113,
	CUDA_FAIL_FFTSHIFT = -114,
	CUDA_FAIL_IFFTSHIFT = -115,
	CUDA_FAIL_TRANSLATE_2D = -116,
	CUDA_FAIL_SET_COMPLEX_REAL_AS_AMPLITUDE = -117,
	CUDA_FAIL_MAKE_BITMASK_2D = -118,
	CUDA_FAIL_COMPARE_BITMASK_2D = -119,
	CUDA_FAIL_MULTIPLY_HADAMARD_2D = -120,
	CUDA_FAIL_DIVIDE_BY_REAL_COMPONENT_2D = -121,
	CULA_INITIALISATION_ERROR = -200,
	CULA_ZGELS_ERROR = -201,
	CULA_ZGETRANSPOSECONJUGATE_ERROR = -202
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
	{ CINPUT_UNRECOGNISED_STAGE, "CINPUT_UNRECOGNISED_STAGE" },
	{ CPROCESS_UNKNOWN_STAGE, "CPROCESS_UNKNOWN_STAGE" },
	{ CPROCESS_REVERT_CROP_REGIONS_INVALID, "CPROCESS_REVERT_CROP_REGIONS_INVALID" },
	{ CPROCESS_REVERT_GROW_REGIONS_INVALID, "CPROCESS_REVERT_GROW_REGIONS_INVALID" },
	{ CPROCESS_REVERT_RESCALE_REGIONS_INVALID, "CPROCESS_REVERT_RESCALE_REGIONS_INVALID" },
	{ CPROCESS_FIT_AND_SUBTRACT_POLY_FAIL_INTEGRITY_CHECK, "CPROCESS_FIT_AND_SUBTRACT_POLY_FAIL_INTEGRITY_CHECK" },
	{ CUDA_FAIL_SYNCHRONIZE, "CUDA_FAIL_SYNCHRONIZE" },
	{ CUDA_FAIL_MEMCPY_HH, "CUDA_FAIL_MEMCPY_HH" },
	{ CUDA_FAIL_MEMCPY_HD, "CUDA_FAIL_MEMCPY_HD" },
	{ CUDA_FAIL_MEMCPY_DH, "CUDA_FAIL_MEMCPY_DH" },
	{ CUDA_FAIL_MEMCPY_DD, "CUDA_FAIL_MEMCPY_DD" },
	{ CUDA_FAIL_FREE_MEMORY_H, "CUDA_FAIL_FREE_MEMORY_H" },
	{ CUDA_FAIL_FREE_MEMORY_D, "CUDA_FAIL_FREE_MEMORY_D" },
	{ CUDA_FAIL_ALLOCATE_MEMORY_H, "CUDA_FAIL_ALLOCATE_MEMORY_H" },
	{ CUDA_FAIL_ALLOCATE_MEMORY_D, "CUDA_FAIL_ALLOCATE_MEMORY_D" },
	{ CUDA_FAIL_SET_MEMORY_D, "CUDA_FAIL_SET_MEMORY_D" },
	{ CUDA_FFT_FAIL_CREATE_PLAN, "CUDA_FFT_FAIL_CREATE_PLAN" },
	{ CUDA_FFT_FAIL_EXECUTE_PLAN, "CUDA_FFT_FAIL_EXECUTE_PLAN" },
	{ CUDA_FAIL_GET_DEVICE_MEMORY, "CUDA_FAIL_GET_DEVICE_MEMORY" },
	{ CUDA_FAIL_GET_SPAXEL_DATA_2D, "CUDA_FAIL_GET_SPAXEL_DATA_2D" },
	{ CUDA_FAIL_FFTSHIFT, "CUDA_FAIL_FFTSHIFT" },
	{ CUDA_FAIL_IFFTSHIFT, "CUDA_FAIL_IFFTSHIFT" },
	{ CUDA_FAIL_TRANSLATE_2D, "CUDA_FAIL_TRANSLATE_2D" },
	{ CUDA_FAIL_SET_COMPLEX_REAL_AS_AMPLITUDE, "CUDA_FAIL_SET_COMPLEX_REAL_AS_AMPLITUDE" },
	{ CUDA_FAIL_MAKE_BITMASK_2D, "CUDA_FAIL_MAKE_BITMASK_2D" },
	{ CUDA_FAIL_COMPARE_BITMASK_2D, "CUDA_FAIL_COMPARE_BITMASK_2D" },
	{ CUDA_FAIL_MULTIPLY_HADAMARD_2D, "CUDA_FAIL_MULTIPLY_HADAMARD_2D" },
	{ CUDA_FAIL_DIVIDE_BY_REAL_COMPONENT_2D, "CUDA_FAIL_DIVIDE_BY_REAL_COMPONENT_2D" },
	{ CULA_INITIALISATION_ERROR, "CULA_INITIALISATION_ERROR" },
	{ CULA_ZGELS_ERROR, "CULA_ZGELS_ERROR" },
	{ CULA_ZGETRANSPOSECONJUGATE_ERROR, "CULA_ZGETRANSPOSECONJUGATE_ERROR" }
};

void throw_error(errors);