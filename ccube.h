#pragma once

#include <valarray>
#include <vector>
#include <string>

#include "ccomplex.cuh"
#include "cspslice.h"
#include "regions.h"
#include "cmemory.h"
#include "errors.h"

enum ccube_domains {
	SPATIAL = 0,
	FREQUENCY = 1
};

enum ccube_states {
	OK = 0,
	INCONSISTENT = 1
};

class cube {
public:
	cube() {};
	~cube() {};
	ccube_domains domain;
	ccube_states state = OK;	// this keeps track of whether the cube has consistently sized slices.
protected:
	virtual int crop(std::vector<rectangle>) { return 0; };
	virtual cube* deepcopy() { return NULL; };
	virtual std::vector<long> get_dim() { return std::vector<long>(); };
	int clear() { return 0; }
	int rescale(float) { return 0; };
};

class dcube;

class hcube : public cube, public hmemory {
public:
	hcube() {};
	hcube(std::valarray<double>, std::vector<long>, std::vector<double>);
	hcube(std::valarray<Complex>, std::vector<long>, std::vector<double>, ccube_domains);
	hcube(dcube*);
	~hcube();
	std::vector<hspslice*> slices;
	int clear();
	int crop(std::vector<rectangle>);
	hcube* deepcopy();
	std::valarray<double> getDataAsValarray(complex_part);
	int rescale(float);
	int write(complex_part, std::string, bool);
};

class dcube : public cube, public dmemory {
public:
	dcube() {};
	dcube(hcube*);
	~dcube();
	std::vector<dspslice*> slices;
	int clear();
	int crop(std::vector<rectangle>);
	dcube* deepcopy();
	int fft(bool);
	int rescale(float);
};





