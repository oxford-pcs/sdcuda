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
	Complex* p_data;
	rectangle region;
	long n_elements;
	double wavelength;
	virtual int crop(long, long, long, long) { return 0; };
	virtual int grow(long, long, long, long) { return 0; };
};

class hspslice : public spslice {
public:
	hcube* datacube;
	hspslice(hcube*, Complex*, rectangle, double);
	~hspslice() {};
	int crop(long, long, long, long);
};

class dspslice : public spslice {
public:
	dcube* datacube;
	dspslice(dcube*, Complex*, rectangle, double);
	~dspslice() {};
	int crop(long, long, long, long);
	int grow(long, long, long, long);
};


