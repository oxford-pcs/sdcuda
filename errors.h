#pragma once

#include <stdio.h>
#include <stdlib.h>

#include "logger.h"

inline void throw_error(int code) {
	if (code < 0) {
		fprintf(stderr, "\nFATAL:\tencountered error with code %d, exiting.\n", code);
		exit(code);
	} else {
		fprintf(stderr, "\nWARN:\tencountered warning with code %d\n", code);
	}
}

enum ccube_errors {
	CCUBE_OK = 0,
	CCUBE_FAIL_BAD_DOMAIN = -1,
	CCUBE_FAIL_WRITE = -2,
	CCUBE_FAIL_NO_INTEGRITY = -3,
};
