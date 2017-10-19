#include "logger.h"
#include <stdio.h>

void broker_to_stdout(char* msg) {
	fprintf(stdout, "BROKER:STDOUT\t\t%s\n", msg);
}

void broker_to_stderr(char* msg) {
	fprintf(stderr, "BROKER:STDOUT\t\t%s\n", msg);
}

void process_to_stdout(char* msg, int pid) {
	fprintf(stdout, "PROCESS:STDOUT:%d\t%s\n", pid, msg);
}

void process_to_stderr(char* msg, int pid) {
	fprintf(stderr, "PROCESS:STDOUT:%d\t%s\n", pid, msg);
}
