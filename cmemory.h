#pragma once;

#include "ccomplex.cuh"

class memory {
public:
	memory() {};
	~memory() {};
	static void memcpydd(Complex*, Complex*, long);
	static void memcpydh(Complex*, Complex*, long);
	static void memcpyhd(Complex*, Complex*, long);
	static void memcpyhh(Complex*, Complex*, long);
protected:
	virtual void free(Complex*) {};
	virtual Complex* malloc(long, bool) { return NULL; };
	virtual Complex* realloc(Complex*, long, long, bool) { return NULL; };
};

class hmemory : public memory {
public:
	hmemory() {};
	~hmemory() {};
	void free(Complex*);
	Complex* malloc(long, bool);
	Complex* realloc(Complex*, long, long, bool);
};

class dmemory : public memory {
public:
	dmemory() {};
	~dmemory() {};
	void free(Complex*);
	Complex* malloc(long, bool);
	Complex* realloc(Complex*, long, long, bool);
};
