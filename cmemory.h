#pragma once;

#include "ccomplex.cuh"

template <class T>
class memory {
public:
	memory() {};
	~memory() {};
	static void memcpydd(T*, T*, long);
	static void memcpydh(T*, T*, long);
	static void memcpyhd(T*, T*, long);
	static void memcpyhh(T*, T*, long);
protected:
	virtual void free(T*) {};
	virtual T* malloc(long, bool) { return NULL; };
	virtual T* realloc(T*, long, long, bool) { return NULL; };
};

template <class T>
class hmemory : public memory<T> {
public:
	hmemory() {};
	~hmemory() {};
	void free(T*);
	T* malloc(long, bool);
	T* realloc(T*, long, long, bool);
};

template <class T>
class dmemory : public memory<T> {
public:
	dmemory() {};
	~dmemory() {};
	void free(T*);
	T* malloc(long, bool);
	T* realloc(T*, long, long, bool);
};
