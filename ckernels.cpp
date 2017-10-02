#include "ckernels.h"
#include "math.h"
#include "math.h"
#include <stdio.h>

#include <cuda_runtime.h>
#include <cufft.h>


Kernel::~Kernel() {
	Kernel::free(Kernel::p_kcoeffs);
}

int Kernel::memcpyhd(double* dst, double* src, long size) {
	cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to copy memory to device\n");
		return 1;
	}
	return 0;
}

int Kernel::free(double* data) {
	if (data != NULL) {
		cudaFree(data);
		if (cudaGetLastError() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to free memory on device\n");
		}
	}
	return 0;
}

double* Kernel::malloc(long size, bool zero_initialise) {
	double* data = NULL;
	cudaMalloc((void**)&(data), size);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to allocate\n");
	}
	if (zero_initialise) {
		cudaMemset(data, 0, size);
		if (cudaGetLastError() != cudaSuccess) {
			fprintf(stderr, "Cuda error: Failed to memset\n");
		}
	}
	return data;
}

LanczosShift::LanczosShift(long a, double shift_x, double shift_y) {
	LanczosShift::ksize = (2 * a) + 1;
	LanczosShift::p_kcoeffs = malloc(LanczosShift::ksize*LanczosShift::ksize*sizeof(Complex), true);
	LanczosShift::shift_x = shift_x;
	LanczosShift::shift_y = shift_y;
	LanczosShift::makeKernel();
}

int LanczosShift::makeKernel() {
	double* h_kcoeffs = new double[LanczosShift::ksize*LanczosShift::ksize];
	long kernel_half_size = (LanczosShift::ksize - 1) / 2;
	for (int j = -kernel_half_size; j <= kernel_half_size; j++) {
		for (int i = -kernel_half_size; i <= kernel_half_size; i++) {
			double kx = nsinc(LanczosShift::shift_x - i) * nsinc((LanczosShift::shift_x - i) / kernel_half_size);
			double ky = nsinc(LanczosShift::shift_y - j) * nsinc((LanczosShift::shift_y - i) / kernel_half_size);
			h_kcoeffs[(i + kernel_half_size) + (j + kernel_half_size)*LanczosShift::ksize] = kx*ky;
		}
	}
	Kernel::memcpyhd(Kernel::p_kcoeffs, h_kcoeffs, LanczosShift::ksize*LanczosShift::ksize*sizeof(double));
	delete h_kcoeffs;
	return 0;
};
