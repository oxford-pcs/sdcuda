#pragma once

#include <vector>

#include "ccomplex.cuh"
#include "regions.h"

class cube;

class spslice {
public:
	spslice(cube*, Complex*, rectangle, double);
	~spslice() {};
	cube* datacube;
	Complex* p_data;
	rectangle region;
	long n_elements;
	double wavelength;
	int crop(long, long, long, long);
};