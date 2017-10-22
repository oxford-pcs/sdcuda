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

hcube* go(input* iinput, int exp_idx) {
	process p(iinput, exp_idx);
	p.run();
	return p.h_datacube->deepcopy();
}

int main(int argc, char **argv) {
	print_banner();

	// Parse the command line input and process
	//
	clparser* iclparser = new clparser(argc, argv);
	input* iinput = new input(iclparser->in_FITS_filename, iclparser->in_params_filename, iclparser->in_config_filename, true);
	printf("\n");

	char broker_message_buffer[255];
	to_stdout("\tBROKER\tstarting asynchronous process broker...");
	std::vector<std::future<hcube*>> running_processes;
	for (int i = 0; i < iinput->dim[2]; i++) {
		int available_slots = iinput->nCPUCORES;
		for (std::vector<std::future<hcube*>>::iterator it = running_processes.begin(); it != running_processes.end(); ++it) {
			if (it->wait_for(std::chrono::microseconds(1)) != future_status::ready) {
				available_slots--;
			}
		}
		sprintf(broker_message_buffer, "\tBROKER\t%d new slot(s) available", available_slots);
		to_stdout(broker_message_buffer);
		if (available_slots > 0) {
			sprintf(broker_message_buffer, "\tBROKER\tassigning new process (%d) to slot", i);
			to_stdout(broker_message_buffer);
			running_processes.push_back(std::async(go, iinput, i));
		} else {
			to_stdout("\tBROKER\twaiting for next available slot");
			bool slot_is_available = false;
			while (!slot_is_available) {
				for (std::vector<std::future<hcube*>>::iterator it = running_processes.begin(); it != running_processes.end(); ++it) {
					if (it->wait_for(std::chrono::milliseconds(1000)) == future_status::ready) {
						to_stdout("\tBROKER\tnew slot available");
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
