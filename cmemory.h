#pragma once;

template <class T>
class memory {
public:
	memory() {};
	~memory() {};
	static void memcpydd(T*, T*, long);
	static void memcpydh(T*, T*, long);
	static void memcpyhd(T*, T*, long);
	static void memcpyhh(T*, T*, long);
};

template <class T>
class hmemory : public memory<T> {
public:
	hmemory() {};
	~hmemory() {};
	static void free(T*);
	static T* malloc(long, bool);
	static T* realloc(T*, long, long, bool);
};

template <class T>
class dmemory : public memory<T> {
public:
	dmemory() {};
	~dmemory() {};
	static void free(T*);
	static T* malloc(long, bool);
	static T* realloc(T*, long, long, bool);
};
