#include "cudacalls.cuh"
#include "ccube.h"

cudaError cudaCompareArray2D(int nCUDABLOCKS, int nCUDATHREADSPERBLOCK, int** in, int* out, long index, long n_slices, long n_spaxels_per_slice) {
	cCompareArray2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(in, out, index, n_slices, n_spaxels_per_slice);
	return cudaGetLastError();
}

cudaError cudaDivideByRealComponent2D(int nCUDABLOCKS, int nCUDATHREADSPERBLOCK, Complex* in1, Complex* in2, Complex* out, long n_spaxels_per_slice) {
	cDivideByRealComponent2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(in1, in2, out, n_spaxels_per_slice);
	return cudaGetLastError();
}

cudaError cudaGetSpaxelData2D(int nCUDABLOCKS, int nCUDATHREADSPERBLOCK, Complex** in, Complex** out, long n_slices, long n_spaxels_per_slice) {
	cGetSpaxelData2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(in, out, n_slices, n_spaxels_per_slice);
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

cudaError cudaMakeBitmask2D(int nCUDABLOCKS, int nCUDATHREADSPERBLOCK, Complex** in, int** out, long n_slices, long n_spaxels_per_slice) {
	cMakeBitmask2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(in, out, n_slices, n_spaxels_per_slice);
	return cudaGetLastError();
}

cudaError cudaMultiplyHadamard2D(int nCUDABLOCKS, int nCUDATHREADSPERBLOCK, Complex* in1, Complex* in2, Complex* out, long n_spaxels_per_slice) {
	cMultiplyHadamard2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(in1, in2, out, n_spaxels_per_slice);
	return cudaGetLastError();
}

cudaError cudaScale2D(int nCUDABLOCKS, int nCUDATHREADSPERBLOCK, Complex* in, double factor, long memsize) {
	cScale2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(in, factor, memsize);
	return cudaGetLastError();
}

cudaError cudaSetComplexRealAsAmplitude2D(int nCUDABLOCKS, int nCUDATHREADSPERBLOCK, Complex* in, long size) {
	cSetComplexRealAsAmplitude2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(in, size);
	return cudaGetLastError();
}

cudaError cudaPolySub2D(int nCUDABLOCKS, int nCUDATHREADSPERBLOCK, Complex** in, int** mask, Complex** coeffs, long n_coeffs, int* wavelengths, long n_slices, long n_spaxels_per_slice) {
	cPolySub2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(in, mask, coeffs, n_coeffs, wavelengths, n_slices, n_spaxels_per_slice);
	return cudaGetLastError();
}

cudaError cudaTranslate2D(int nCUDABLOCKS, int nCUDATHREADSPERBLOCK, Complex* in, double2 translation, long x_size) {
	cTranslate2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(in, translation, x_size);
	return cudaGetLastError();
}
