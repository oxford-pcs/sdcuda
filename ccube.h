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
	std::vector<spslice> slices;
	Complex *p_data = NULL;						// pointer to data block, not necessarily with contiguous data. always use member [slices]
	int crop(std::vector<rectangle>);
	int memcpydd(Complex*, Complex*, long);
	int memcpydh(Complex*, Complex*, long);
	int memcpyhd(Complex*, Complex*, long);
	int memcpyhh(Complex*, Complex*, long);
	rectangle rescale(float);
protected:
	virtual int clear();
	virtual cube* copy();
	virtual int free(Complex*);
	virtual Complex* malloc(long, bool);
};

class dcube;

class hcube : public cube {
public:
	hcube() {};
	hcube(std::valarray<double>, std::vector<long>, std::vector<double>);
	hcube(dcube*);
	~hcube();
	hcube* copy();
	std::valarray<double> getDataAsValarray(complex_part);
	int clear();
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
	dcube* copy();
	int clear();
	int fft(bool);
private:
	int free(Complex*);
	Complex* malloc(long, bool);

};





