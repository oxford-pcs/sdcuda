#pragma once

#include <vector>
#include <valarray>

#include "ccomplex.cuh"
#include "regions.h"
#include "cmemory.h"

class spslice {
public:
	spslice() {};
	~spslice() {};
	Complex* p_data = NULL;
	rectangle region;
	long memsize;
	int wavelength;			// nm
	long2 getDimensions();
	long getNumberOfElements();
	virtual int clear() { return 0; };
	virtual int crop(rectangle) { return 0; };
	virtual int grow(rectangle) { return 0; };
};

class hcube;
class dspslice;

class hspslice : public spslice, public hmemory {
public:
	hcube* datacube;
	hspslice() {};
	hspslice(hcube*, std::valarray<double>, rectangle, int);
	hspslice(hcube*, std::valarray<Complex>, rectangle, int);
	hspslice(hcube*, dspslice*);
	~hspslice();
	int clear();
	int crop(rectangle);
	hspslice* deepcopy();
	int grow(rectangle);
};

class dcube;

class dspslice : public spslice, public dmemory {
public:
	dcube* datacube;
	dspslice() {};
	dspslice(dcube*, std::valarray<double>, rectangle, int);
	dspslice(dcube*, std::valarray<Complex>, rectangle, int);
	dspslice(dcube*, hspslice*);
	~dspslice();
	int clear();
	int crop(rectangle);
	dspslice* deepcopy();
	int grow(rectangle);
};


