#pragma once

#include "ccomplex.h"
#include "cdevice.h"
#include "ccube.h"

cudaError cudaSubtractPoly(int, int, Complex**, Complex*, long, int*, long, long);
cudaError cudaGetSpaxelData2D(int, int, Complex**, Complex*, long, long);
cudaError cudaFftShift2D(int, int, Complex*, Complex*, long);
cudaError cudaIFftShift2D(int, int, Complex*, Complex*, long);
cudaError cudaScale2D(int, int, Complex*, double, long);
cudaError cudaSetComplexRealAsAmplitude(int, int, Complex*, long);
cudaError cudaTranslate2D(int, int, Complex*, double2, long);