#pragma once;

#include "ccomplex.cuh"

class memory {
public:
	memory() {};
	~memory() {};
	int memcpydd(Complex*, Complex*, long);
	int memcpydh(Complex*, Complex*, long);
	int memcpyhd(Complex*, Complex*, long);
	int memcpyhh(Complex*, Complex*, long);
protected:

	virtual int free(Complex*) { return 0; };
	virtual Complex* malloc(long, bool) { return NULL; };
};

class hmemory : public memory {
public:
	hmemory() {};
	~hmemory() {};
	int free(Complex*);
	Complex* malloc(long, bool);
};

class dmemory : public memory {
public:
	dmemory() {};
	~dmemory() {};
	int free(Complex*);
	Complex* malloc(long, bool);
};
