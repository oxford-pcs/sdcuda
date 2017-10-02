#pragma once

#include <vector>

#include "ccomplex.cuh"
#include "ccube.h"
#include "cmemory.h"

using namespace std;

class spaxel {
public:
	spaxel() {};
	~spaxel() {};
	std::vector<double> wavelengths;
	Complex* p_data = NULL;
	long memsize;
	long n_elements;
};

class hspaxel : public spaxel, public hmemory {
	hspaxel() {};
	~hspaxel() {};
};

class dspaxel : public spaxel, public dmemory {
	dspaxel(dcube*);
	~dspaxel() {};
};