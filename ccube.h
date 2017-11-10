#pragma once

#include <valarray>
#include <vector>
#include <string>

#include "ccomplex.h"
#include "cdevice.cuh"
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
	virtual void clear() {};
	virtual void crop(rectangle) {}
	virtual void crop(std::vector<rectangle>) {};
	virtual cube* deepcopy() { return NULL; };
	virtual rectangle getLargestSliceRegion() { return rectangle(); };
	virtual int getNumberOfSpaxels() { return 0; }
	virtual rectangle getSmallestSliceRegion() { return rectangle(); };
	void grow(rectangle) {};
	void grow(std::vector<rectangle>) {};
	virtual void rescale(float) {};
};

class dcube;
class hcube : public cube, public hmemory<Complex> {
public:
	hcube() {};
	hcube(std::valarray<double>, std::vector<long>, std::vector<int>);
	hcube(std::valarray<Complex>, std::vector<long>, std::vector<int>, ccube_domains);
	hcube(dcube*);
	~hcube();
	std::vector<hspslice*> slices;
	void clear();
	void crop(rectangle);
	void crop(std::vector<rectangle>);
	hcube* deepcopy();
	std::valarray<double> getDataAsValarray(complex_part);
	std::valarray<double> getDataAsValarray(complex_part, int);
	rectangle getLargestSliceRegion();
	int getNumberOfSpaxels();
	rectangle getSmallestSliceRegion();
	void grow(rectangle);
	void grow(std::vector<rectangle>);
	void rescale(std::vector<double>);
	void write(complex_part, std::string, bool);
	void write(complex_part, std::string, int, bool);
};

class dcube : public cube, public dmemory<Complex> {
public:
	dcube() {};
	dcube(hcube*);
	~dcube();
	std::vector<dspslice*> slices;
	void clear();
	void crop(rectangle);
	void crop(std::vector<rectangle>);
	dcube* deepcopy();
	void fft(bool);
	rectangle getLargestSliceRegion();
	int getNumberOfSpaxels();
	rectangle getSmallestSliceRegion();
	void grow(rectangle);
	void grow(std::vector<rectangle>);
	void rescale(std::vector<double>);
};





