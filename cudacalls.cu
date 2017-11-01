#include "cudacalls.cuh"
#include "ccube.h"

cudaError cudaSubtractPoly(int nCUDABLOCKS, int nCUDATHREADSPERBLOCK, Complex** in, Complex* coeffs, long n_coeffs, int* wavelengths, long n_slices, long n_spaxels) {
	cSubtractPoly << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(in, coeffs, n_coeffs, wavelengths, n_slices, n_spaxels);
	return cudaGetLastError();
}

cudaError cudaGetSpaxelData2D(int nCUDABLOCKS, int nCUDATHREADSPERBLOCK, Complex** in, Complex* out, long n_slices, long n_spaxels) {
	cGetSpaxelData2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(in, out, n_slices, n_spaxels);
	return cudaGetLastError();
}

cudaError cudaFftShift2D(int nCUDABLOCKS, int nCUDATHREADSPERBLOCK, Complex* in, Complex* out, long x_size) {
	cFftShift2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(in, out, x_size);
	return cudaGetLastError();
}

cudaError cudaIFftShift2D(int nCUDABLOCKS, int nCUDATHREADSPERBLOCK, Complex* in, Complex* out, long x_size) {
	cIFftShift2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(in, out, x_size);
	return cudaGetLastError();
}

cudaError cudaScale2D(int nCUDABLOCKS, int nCUDATHREADSPERBLOCK, Complex* data, double constant, long memsize) {
	cScale2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(data, constant, memsize);
	return cudaGetLastError();
}

cudaError cudaSetComplexRealAsAmplitude(int nCUDABLOCKS, int nCUDATHREADSPERBLOCK, Complex* a, long size) {
	cSetComplexRealAsAmplitude << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(a, size);
	return cudaGetLastError();
}

cudaError cudaTranslate2D(int nCUDABLOCKS, int nCUDATHREADSPERBLOCK, Complex* a, double2 translation, long x_size) {
	cTranslate2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(a, translation, x_size);
	return cudaGetLastError();
}
