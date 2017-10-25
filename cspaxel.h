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
	Complex** p_data = NULL;
};

class hspaxel : public spaxel, public hmemory<Complex*> {
public:
	hspaxel(hcube*, int);
	~hspaxel() {};
	hcube* h_datacube;
};

class dspaxel : public spaxel, public dmemory<Complex*> {
public:
	dspaxel(dcube*, int);
	~dspaxel() {};
	dcube* d_datacube;
};