#include "logger.h"
#include <stdio.h>
#include <ctime>

char* get_timestamp() {
	time_t now = time(0);
	tm *ltm = localtime(&now);
	char buf[100];
	strftime(buf, 100, "%m-%d-%Y %H:%M:%S", ltm);
	return buf;
}

void to_stdout(char* msg) {
	fprintf(stdout, "%s\t%s\n", get_timestamp(), msg);
}

void to_stderr(char* msg) {
	fprintf(stderr, "%s\t%s\n", get_timestamp(), msg);
}
