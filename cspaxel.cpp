#include "cspaxel.h"
#include "cspslice.h"

#include "ccube.h"

hspaxel::hspaxel(hcube* h_datacube, int spaxel_idx) {
	/*
	Construct a spaxel in host memory using spaxel [spaxel_idx] of cube [h_datacube].
	*/
	hspaxel::h_datacube = h_datacube;
	hspaxel::p_data = hspaxel::malloc(hspaxel::h_datacube->slices.size()*sizeof(Complex*), true);
	for (int i = 0; i < hspaxel::h_datacube->slices.size(); i++) {
		hspaxel::p_data[i] = hspaxel::h_datacube->slices[i]->p_data[spaxel_idx];
	}
}

hspaxel::~hspaxel() {
	hspaxel::free(hspaxel::p_data);
}

dspaxel::dspaxel(dcube* d_datacube, int spaxel_idx) {
	/*
	Construct a spaxel in device memory using spaxel [spaxel_idx] of cube [d_datacube].
	*/
	dspaxel::d_datacube = d_datacube;
	dspaxel::p_data = dspaxel::malloc(dspaxel::d_datacube->slices.size()*sizeof(Complex*), true);
	printf("%d\n", dspaxel::d_datacube->slices.size()*sizeof(Complex*));
	for (int i = 0; i < dspaxel::d_datacube->slices.size(); i++) {
		dspaxel::p_data[i] = dspaxel::d_datacube->slices[i]->p_data[spaxel_idx];
	}
	exit(0);
}

dspaxel::~dspaxel() {
	dspaxel::free(dspaxel::p_data);
}