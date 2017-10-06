#pragma once

#include <stdio.h>

enum process_states {
	PROCESS_STARTED = 0,
	COPIED_DATACUBE_TO_HOST = 1,
	CROPPED_TO_SQUARE = 2,
	COPIED_DATACUBE_TO_DEVICE = 3,
	FFTED = 4,
	NORMALISED = 5,
	FFTSHIFTED = 6,
	RESCALED = 7,
	IFFTSHIFTED = 8,
	IFFTED = 9,
	CROPPED_TO_SMALLEST_DIMENSION = 10,
	APPLIED_LANCZOS = 11
};

process_states& operator++(process_states& state) {
	state = static_cast<process_states>(static_cast<int>(state)+1);
	return state;
}

struct process_monitor {
	process_states state;
	process_monitor(process_states start) {
		process_monitor::state = start;
	};
	int next() {
		process_monitor::state++;
		printf("MSG:\tprocess state changed to %d\n", state);
		return state;
	};
};