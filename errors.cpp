#include "errors.h"

void throw_error(errors code) {
	if (code < 0) {
		char buf[255];
		sprintf(buf, "\tFATAL\t\tencountered error with code %d (%s), exiting.\n", code, error_messages.at(code));
		to_stderr(buf);
		exit(code);
	} else {
		char buf[255];
		sprintf(buf, "\tWARN\t\tencountered error with code %d (%s), exiting.\n", code, error_messages.at(code));
		to_stderr(buf);
	}
}
