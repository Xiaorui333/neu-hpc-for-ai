#pragma once

#include "color.h"
#include "time.h"

void sleep_seconds(int seconds) {
    yellow("sleep begin (%d sec)", seconds);

    struct timespec ts;
    ts.tv_sec = seconds;      // seconds
    ts.tv_nsec = 0;           // nanoseconds
    nanosleep(&ts, NULL);

    yellow("sleep end");
}
