#pragma once

#include "ccomplex.h"
#include "cdevice.cuh"
#include "ccube.h"

cudaError cudaCompareArray2D(int, int, int**, int*, long, long, long);
cudaError cudaDivideByRealComponent2D(int, int, Complex*, Complex*, Complex*, long);
cudaError cudaGetSpaxelData2D(int, int, Complex**, Complex**, long, long);
cudaError cudaFftShift2D(int, int, Complex*, Complex*, long);
cudaError cudaIFftShift2D(int, int, Complex*, Complex*, long);
cudaError cudaMakeBitmask2D(int, int, Complex**, int**, long, long);
cudaError cudaMultiplyHadamard2D(int, int, Complex*, Complex*, Complex*, long);
cudaError cudaScale2D(int, int, Complex*, double, long);
cudaError cudaSetComplexRealAsAmplitude2D(int, int, Complex*, long);
cudaError cudaPolySub2D(int, int, Complex**, int**, Complex**, long, int*, long, long);
cudaError cudaTranslate2D(int, int, Complex*, double2, long);