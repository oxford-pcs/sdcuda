#include "cspaxel.h"

#include "ccube.h"

dspaxel::dspaxel(dcube* d_datacube, std::vector<long> idx) {
	/*
	Construct a spaxel in device memory using a cube [d_datacube] in device memory.
	*/
	dspaxel::d_datacube = d_datacube;
	dspaxel::p_data = dspaxel::malloc(d_datacube->dim[2] * sizeof(Complex), true);	// this is an array of pointers pointing towards d_datacube [p_data] values
}

dspaxel::~dspaxel() {
	dspaxel::free(dspaxel::p_data);
}