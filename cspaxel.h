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
	Complex* p_data = NULL;
	long memsize;
	long n_elements;
};

class hspaxel : public spaxel, public hmemory {
public:
	hspaxel() {};
	~hspaxel() {};
	hcube* h_datacube;
};

class dspaxel : public spaxel, public dmemory {
public:
	dspaxel(dcube*, std::vector<long>);
	~dspaxel();
	dcube* d_datacube;
};