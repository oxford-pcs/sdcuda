#pragma once

#include <valarray>
#include <vector>
#include <string>

#include "ccomplex.cuh"
#include "cspslice.h"
#include "regions.h"
#include "cmemory.h"

enum domains {
	SPATIAL = 0,
	FREQUENCY = 1
};

enum ccube_errors {
	CCUBE_OK = 0,
	CCUBE_FFT_BAD_DOMAIN = -1,
};

inline void throw_error(int code) {
	if (code < 0) {
		fprintf(stderr, "\nccube: fatal error encountered with code: %d, exiting.\n", code);
		exit(0);
	} else {
		fprintf(stderr, "\nccube: warning encountered with code: %d\n", code);
	}
}

class cube {
public:
	cube() {};
	~cube() {};
	std::vector<long> dim;
	domains domain;
	long memsize;
	long n_elements;
	std::vector<double> wavelengths;
	Complex *p_data = NULL;						// pointer to data block, not necessarily with contiguous data. always use member [slices]
protected:
	virtual int clear() { return 0; };
	virtual cube* copy() { return NULL; };
	virtual int crop(std::vector<rectangle>) { return 0; };
	int rescale(float, rectangle&) { return 0; };
};

class dcube;

class hcube : public cube, public hmemory {
public:
	hcube() {};
	hcube(std::valarray<double>, std::vector<long>, std::vector<double>);
	hcube(std::valarray<Complex>, std::vector<long>, std::vector<double>, domains);
	hcube(dcube*);
	~hcube();
	hcube* copy();
	std::vector<hspslice> slices;
	std::valarray<double> getDataAsValarray(complex_part);
	int clear();
	int crop(std::vector<rectangle>);
	int rescale(float, rectangle&);
	int write(complex_part, std::string, bool);
};

class dcube : public cube, public dmemory {
public:
	dcube() {};
	dcube(hcube*);
	~dcube() {};
	std::vector<dspslice> slices;
	dcube* copy();
	int clear();
	int crop(std::vector<rectangle>);
	int fft(bool);
	int rescale(float, rectangle&);
};





