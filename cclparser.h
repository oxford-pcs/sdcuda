#pragma once

#include <iostream>

enum clparser_states {
	CCLPARSER_NOT_CHECKED = 2,
	CCLPARSER_NOT_PARSED = 1,
	CCLPARSER_OK = 0,
	CCLPARSER_FAULT_MISSING_ARGUMENT = -1,
	CCLPARSER_FAULT_UNKNOWN_OPTION = -2,
	CCLPARSER_FAULT_UNKNOWN_CHARACTER = -3,
	CCLPARSER_FAULT_INVALID_NUMBER_OF_PARAMETERS = -4,
	CCLPASRER_FAULT_INPUT_FITS_FILE_NO_EXIST = -5,
	CCLPARSER_FAULT_INPUT_PARAMETERS_FILE_NO_EXIST = -6,
	CCLPARSER_FAULT_UNKNOWN_CONDITION = -99
};

inline bool is_file_existing(const std::string& name) {
	if (FILE *file = fopen(name.c_str(), "r")) {
		fclose(file);
		return true;
	}
	else {
		return false;
	}
}

class clparser {
public:
	clparser(int, char**);
	~clparser() {};
	std::string in_FITS_filename = "";
	std::string in_params_filename = "";
	std::string out_FITS_filename = "";
	int nargs = 0;
	clparser_states state = CCLPARSER_NOT_PARSED;
private:
	int argc;
	char** argv;
	bool check();
	bool parse();
};