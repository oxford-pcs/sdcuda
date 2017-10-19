#pragma once

#include <stdio.h>

struct rectangle {
public:
	long x_start, y_start, x_size, y_size;
	rectangle() {
		x_size = 0;
		y_size = 0;
	};
	rectangle(long, long, long, long);
	~rectangle() {};
	rectangle operator-(const rectangle&);
};