#pragma once

#include "ccomplex.cuh"

#define M_PI	3.14159265358979323846

inline double nsinc(double x) {
	if (x == 0) {
		return 1;
	}
	else {
		return sin(M_PI*x) / (M_PI*x);
	}
}

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