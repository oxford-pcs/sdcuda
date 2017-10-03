#pragma once;

#include "ccomplex.cuh"

class memory {
public:
	memory() {};
	~memory() {};
	static int memcpydd(Complex*, Complex*, long);
	static int memcpydh(Complex*, Complex*, long);
	static int memcpyhd(Complex*, Complex*, long);
	static int memcpyhh(Complex*, Complex*, long);
protected:
	virtual int free(Complex*) { return 0; };
	virtual Complex* malloc(long, bool) { return NULL; };
	virtual Complex* realloc(Complex*, long, long, bool) { return NULL; };
};

class hmemory : public memory {
public:
	hmemory() {};
	~hmemory() {};
	int free(Complex*);
	Complex* malloc(long, bool);
	Complex* realloc(Complex*, long, long, bool);
};

class dmemory : public memory {
public:
	dmemory() {};
	~dmemory() {};
	int free(Complex*);
	Complex* malloc(long, bool);
	Complex* realloc(Complex*, long, long, bool);
};
