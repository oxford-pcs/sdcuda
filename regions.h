#pragma once


#include <stdio.h>
class rectangle {
public:
	rectangle() {};
	rectangle(long, long, long, long);
	~rectangle() {};
	rectangle operator-(const rectangle&);
	long x_start, y_start, x_size, y_size;
};