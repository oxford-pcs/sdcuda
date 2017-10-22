#pragma once

#include "ccomplex.cuh"

void cudaFftShift2D(int, int, Complex*, Complex*, long);
void cudaIFftShift2D(int, int, Complex*, Complex*, long);
void cudaScale2D(int, int, Complex*, double, long);
void cudaSetComplexRealAsAmplitude(int, int, Complex*, long);
void cudaTranslate2D(int, int, Complex*, double2, long);