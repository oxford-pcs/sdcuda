#pragma once

#include <iostream>

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
	std::string in_config_filename = "";
	std::string out_FITS_filename = "";
	int nargs = 0;
private:
	int argc;
	char** argv;
	bool check();
	bool parse();
};