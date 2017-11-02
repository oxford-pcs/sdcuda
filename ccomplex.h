#pragma once

struct Complex {
	double x, y;
	Complex() {};
	Complex(double x, double y) {
		Complex::x = x;
		Complex::y = y;
	}
};

enum complex_part {
	REAL,
	IMAGINARY,
	AMPLITUDE,
	PHASE
};
