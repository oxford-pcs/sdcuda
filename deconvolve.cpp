#include <future>
#include "getopt.h"
#include <thread>
#include <chrono>

#include "cclparser.h"
#include "cinput.h"
#include "cprocess.h"
#include "logger.h"
#include "banner.h"

using namespace std;

process* go(input* iinput, int exp_idx) {
	process* p = new process(iinput, exp_idx);
	p->run();
	return p;
}

int main(int argc, char **argv) {
	print_banner();

	clparser* iclparser = new clparser(argc, argv);
	input* iinput = new input(iclparser->in_FITS_filename, iclparser->in_params_filename, iclparser->in_config_filename, true);
	printf("\n");

	char broker_message_buffer[255];
	to_stdout("\tBROKER\t\tstarting asynchronous process broker...");
	std::vector<std::future<process*>> running_processes;
	int exposure = 0;
	while (exposure < iinput->dim[2]) {
		int available_slots = stoi(iinput->config_host["nCPUCORES"]);
		for (std::vector<std::future<process*>>::iterator it = running_processes.begin(); it != running_processes.end(); ++it) {
			if (it->wait_for(std::chrono::microseconds(1)) != future_status::ready) {
				available_slots--;
			}
		}
		sprintf(broker_message_buffer, "\tBROKER\t\t%d new slot(s) available", available_slots);
		to_stdout(broker_message_buffer);
		if (available_slots > 0) {
			sprintf(broker_message_buffer, "\tBROKER\t\tassigning new process (%d) to slot", exposure);
			to_stdout(broker_message_buffer);
			running_processes.push_back(std::async(go, iinput, exposure));
			exposure++;
		} else {
			to_stdout("\tBROKER\t\twaiting for next available slot");
			bool slot_is_available = false;
			while (!slot_is_available) {
				for (std::vector<std::future<process*>>::iterator it = running_processes.begin(); it != running_processes.end(); ++it) {
					if (it->wait_for(std::chrono::milliseconds(1000)) == future_status::ready) {
						process* p = it->get();
						delete p;
						to_stdout("\tBROKER\t\tnew slot available");
						running_processes.erase(it);
						slot_is_available = true;
						break;
					}
				}
			}
		}
	}
	// make sure last processes complete
	for (std::vector<std::future<process*>>::iterator it = running_processes.begin(); it != running_processes.end(); it++) {
		process* p = it->get();
		p->h_datacube->write(AMPLITUDE, iclparser->out_FITS_filename, true); // FIXME: need to construct 4d cube! should be storing datacubes from above async?
		delete p;
	}
	delete iinput;
	delete iclparser;
	
	exit(EXIT_SUCCESS);
}
