#pragma once

enum states {
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

states& operator++(states& state) {
	state = static_cast<states>(static_cast<int>(state) + 1);
	printf("MSG:\tprocess state changed to %d\n", state);
	return state;
}