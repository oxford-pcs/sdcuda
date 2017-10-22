#include "errors.h"

void throw_error(errors code) {
	if (code < 0) {
		char buf[255];
		sprintf(buf, "FATAL\t\tencountered error with code %d (%s), exiting.", code, error_messages.at(code));
		to_stderr(buf);
		exit(code);
	} else {
		char buf[255];
		sprintf(buf, "WARN\t\tencountered error with code %d (%s), exiting.", code, error_messages.at(code));
		to_stderr(buf);
	}
}
