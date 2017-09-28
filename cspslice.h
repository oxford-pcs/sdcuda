#pragma once

#include <vector>

#include "ccomplex.cuh"
#include "regions.h"

class cube;
class dcube;
class hcube;

class spslice {
public:
	spslice() {};
	~spslice() {};
	cube* datacube;
	Complex* p_data;
	rectangle region;
	long n_elements;
	double wavelength;
	virtual int crop(long, long, long, long) { return 0; };
};

class hspslice : public spslice {
public:
	hspslice(hcube*, Complex*, rectangle, double);
	~hspslice() {};
	int crop(long, long, long, long);
};

class dspslice : public spslice {
public:
	dspslice(dcube*, Complex*, rectangle, double);
	~dspslice() {};
	int crop(long, long, long, long);
};


