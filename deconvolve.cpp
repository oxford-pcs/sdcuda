#include <future>
#include "getopt.h"
#include <thread>
#include <chrono>

#include "cclparser.h"
#include "cinput.h"
#include "ccube.h"
#include "ccomplex.cuh"
#include "regions.h"
#include "cspaxel.h"
#include "cprocess.h"
#include "logger.h"
#include "banner.h"

const int nCPUCORES = 4;

std::list<process_stages> STAGES{
	MAKE_DATACUBE_ON_HOST,
	H_CROP_TO_EVEN_SQUARE,
	COPY_HOST_DATACUBE_TO_DEVICE,
	D_FFT,
	D_FFTSHIFT,
	D_RESCALE,
	D_IRESCALE,
	D_IFFTSHIFT,
	D_IFFT,
	D_SET_DATA_TO_AMPLITUDE,
	D_CROP_TO_SMALLEST_DIMENSION,
	COPY_DEVICE_DATACUBE_TO_HOST
};

hcube* go(input* iinput, clparser* iclparser, int exp_idx) {
	process p(STAGES, iinput, iclparser, exp_idx);
	p.run();
	return p.h_datacube->deepcopy();
}

int main(int argc, char **argv) {
	print_banner();

	// Parse the command line input
	//
	clparser* iclparser = new clparser(argc, argv);
	if (iclparser->state != CCLPARSER_OK) {
		exit(EXIT_FAILURE);
	}

	// Process the input files parsed from the command line input
	//
	input* iinput = new input(iclparser->in_FITS_filename, iclparser->in_params_filename, true);
	if (iinput->state != CINPUT_OK) {
		exit(EXIT_FAILURE);
	}
	printf("\n");

	broker_to_stdout("starting asynchronous process broker...");
	std::vector<std::future<hcube*>> running_processes;
	for (int i = 0; i < iinput->dim[2]; i++) {
		int available_slots = nCPUCORES;
		for (std::vector<std::future<hcube*>>::iterator it = running_processes.begin(); it != running_processes.end(); ++it) {
			if (it->wait_for(std::chrono::microseconds(1)) != future_status::ready) {
				available_slots--;
			}
		}
		char buf[100]; sprintf(buf, "%d new slot(s) available", available_slots);
		broker_to_stdout(buf);
		if (available_slots > 0) {
			char buf[100]; sprintf(buf, "assigning new process (%d) to slot", i);
			broker_to_stdout(buf);
			running_processes.push_back(std::async(go, iinput, iclparser, i));
		} else {
			broker_to_stdout("waiting for next available slot");
			bool slot_is_available = false;
			while (!slot_is_available) {
				for (std::vector<std::future<hcube*>>::iterator it = running_processes.begin(); it != running_processes.end(); ++it) {
					if (it->wait_for(std::chrono::milliseconds(1000)) == future_status::ready) {
						broker_to_stdout("new slot available");
						running_processes.erase(it);
						slot_is_available = true;
						break;
					}
				}
			}
		}
	}
	// make sure last processes complete
	for (std::vector<std::future<hcube*>>::iterator it = running_processes.begin(); it != running_processes.end(); it++) {
		hcube* h = it->get();
		h->write(AMPLITUDE, iclparser->out_FITS_filename, true);		// FIXME: need to construct 4d cube! should be storing datacubes from above async?
		delete h;
	}
	delete iinput;
	delete iclparser;
	
	exit(EXIT_SUCCESS);
}
