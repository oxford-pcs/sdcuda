#include "cspaxel.h"
#include "cspslice.h"

#include "ccube.h"

hspaxel::hspaxel(hcube* h_datacube, int spaxel_idx) {
}

dspaxel::dspaxel(dcube* d_datacube, int spaxel_idx) {
	/*
	Construct a spaxel in device memory using spaxel [spaxel_idx] of cube [d_datacube].
	*/
	dspaxel::d_datacube = d_datacube;
	dspaxel::p_data = dspaxel::malloc(dspaxel::d_datacube->slices.size()*sizeof(Complex*), true);
	for (int i = 0; i < dspaxel::d_datacube->slices.size(); i++) {
		dspaxel::p_data[i] = dspaxel::d_datacube->slices[i]->p_data + 1;
	}
	exit(0);

}
