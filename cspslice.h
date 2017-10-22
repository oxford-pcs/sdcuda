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
	virtual void clear() {};
	virtual void crop(rectangle) {};
	virtual void grow(rectangle) {};
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
	void clear();
	void crop(rectangle);
	hspslice* deepcopy();
	void grow(rectangle);
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
	void clear();
	void crop(rectangle);
	dspslice* deepcopy();
	void grow(rectangle);
};


