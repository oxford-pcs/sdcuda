#pragma once

#include "ccomplex.cuh"

class Kernel {
public: 
	Kernel() {};
	~Kernel();
	long ksize;
	double* p_kcoeffs = NULL;
protected:
	int memcpyhd(double*, double*, long);
	int free(double*);
	double* malloc(long, bool);
};

class LanczosShift : public Kernel {
public:
	LanczosShift(long, double, double);
	~LanczosShift() {};
	double shift_x, shift_y;
private:
	int makeKernel();
};