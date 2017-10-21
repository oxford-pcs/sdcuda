#include "cudacalls.cuh"

void cudaConvolveKernelReal2D(Complex* in, Complex* out, long x_size, double* p_kcoeffs, long ksize) {
	cConvolveKernelReal2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(in, out, x_size, p_kcoeffs, ksize);
}

void cudaFftShift2D(Complex* in, Complex* out, long x_size) {
	cFftShift2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(in, out, x_size);
}

void cudaIFftShift2D(Complex* in, Complex* out, long x_size) {
	cIFftShift2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(in, out, x_size);
}

void cudaScale2D(Complex* data, double constant, long memsize) {
	cScale2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(data, constant, memsize);
}

void cudaSetComplexRealAsAmplitude(Complex* a, long size) {
	cSetComplexRealAsAmplitude << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(a, size);
}

void cudaTranslate2D(Complex* a, double2 translation, long x_size) {
	cTranslate2D << <nCUDABLOCKS, nCUDATHREADSPERBLOCK >> >(a, translation, x_size);
}
