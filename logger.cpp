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

void broker_to_stdout(char* msg) {
	fprintf(stdout, "%s\tBROKER\t%s\n", get_timestamp(), msg);
}

void broker_to_stderr(char* msg) {
	fprintf(stderr, "%s\tBROKER\t\t%s\n", get_timestamp(), msg);
}

void process_to_stdout(char* msg, int pid) {
	fprintf(stdout, "%s %d\tPROCESS\t%s\n", get_timestamp(), pid, msg);
}

void process_to_stderr(char* msg, int pid) {
	fprintf(stderr, "%s %d\tPROCESS\t%s\n", get_timestamp(), pid, msg);
}
