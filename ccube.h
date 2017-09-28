#pragma once

#include <valarray>
#include <vector>
#include <string>

#include "ccomplex.cuh"
#include "cspslice.h"
#include "regions.h"

class cube {
public:
	cube() {};
	~cube() {};
	std::vector<long> dim;
	long memsize;
	long n_elements;
	std::vector<double> wavelengths;
	Complex *p_data = NULL;						// pointer to data block, not necessarily with contiguous data. always use member [slices]
	int memcpydd(Complex*, Complex*, long);
	int memcpydh(Complex*, Complex*, long);
	int memcpyhd(Complex*, Complex*, long);
	int memcpyhh(Complex*, Complex*, long);
protected:
	virtual int clear() { return 0; };
	virtual cube* copy() { return NULL; };
	virtual int crop(std::vector<rectangle>) { return 0; };
	virtual int free(Complex*) { return 0; };
	virtual Complex* malloc(long, bool) { return NULL; };
	virtual rectangle rescale(float) { return rectangle(); };
};

class dcube;

class hcube : public cube {
public:
	hcube() {};
	hcube(std::valarray<double>, std::vector<long>, std::vector<double>);
	hcube(dcube*);
	~hcube();
	hcube* copy();
	std::vector<hspslice> slices;
	std::valarray<double> getDataAsValarray(complex_part);
	int clear();
	int crop(std::vector<rectangle>);
	rectangle rescale(float);
	int write(complex_part, std::string, bool);
private:
	int free(Complex*);
	Complex* malloc(long, bool);

};

class dcube : public cube {
public:
	dcube() {};
	dcube(hcube*);
	~dcube() {};
	std::vector<dspslice> slices;
	dcube* copy();
	int clear();
	int crop(std::vector<rectangle>);
	int fft(bool);
	rectangle rescale(float);
private:
	int free(Complex*);
	Complex* malloc(long, bool);

};





