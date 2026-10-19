/* rdma_sampler.c — RDMA byte-counter sampler for netinfermon
 *
 *   rdma_sampler [-i us] [-n ticks] [-d dev,dev] [-b sysfs_root] [-c core] [-r] [-o file]
 *
 *   -i  Sampling interval in microseconds (default 1000). If a tick takes longer than the
 *       interval, the next one starts without waiting (no backlog builds up); real timestamps
 *       are always written, so rates are not affected.
 *   -n  Number of ticks (default 0 = run until stopped)
 *   -d  Devices to read, comma-separated (default: every device whose ports/1/state is ACTIVE)
 *   -b  sysfs root (default /sys/class/infiniband; used by the tests)
 *   -c  Pin to a CPU core (default: no pinning)
 *   -r  SCHED_FIFO priority (default: off, so the measured workload is not disturbed)
 *   -o  Output file (default stdout)
 *
 * Output (CSV, one record per line):
 *   H,2,<interval_us>,<device_count>
 *   D,<idx>,<device>
 *   C,<mono_raw_ns>,<realtime_ns>        clock anchor (at start and every ~10 s)
 *   S,<tick>,<idx>,<tx_a>,<tx_b>,<tx>,<rx_a>,<rx_b>,<rx>
 *   E,<tick>,<idx>,<tx|rx>,<errno>        a counter could not be read; no S line is written
 *                                          for that device in that tick (no made-up values)
 *
 * Every counter read is bracketed by its own (a, b) timestamps (CLOCK_MONOTONIC_RAW, ns).
 * The counter was really read at some moment inside [a, b]: the midpoint is the sample time
 * and (b - a) / 2 is its time uncertainty. Values are in 4-byte words (bytes = value x 4).
 * Only the two data counters are read: every read is a firmware query (~0.1-0.3 ms), and
 * unnecessary reads make the tick longer.
 */

#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#define MAXDEV 16

typedef struct {
    char name[64];
    int  fd_tx, fd_rx;
} rdma_dev;

static const char *base = "/sys/class/infiniband";
static rdma_dev devs[MAXDEV];
static int ndev = 0;

static long long ts_ns(const struct timespec *ts) {
    return (long long)ts->tv_sec * 1000000000LL + ts->tv_nsec;
}

static long long clock_ns(clockid_t id) {
    struct timespec ts;
    clock_gettime(id, &ts);
    return ts_ns(&ts);
}

/* 0 = success, otherwise errno. *out is not changed on a failed read. */
static int read_cnt(int fd, long long *out) {
    char buf[64];
    ssize_t n = pread(fd, buf, sizeof buf - 1, 0);
    if (n < 0) return errno ? errno : EIO;
    if (n == 0) return EIO;
    buf[n] = '\0';
    char *end;
    errno = 0;
    long long v = strtoll(buf, &end, 10);
    if (errno) return errno;
    if (end == buf || v < 0) return EINVAL;
    *out = v;
    return 0;
}

static int port_active(const char *dev) {
    char path[512], buf[64];
    snprintf(path, sizeof path, "%s/%s/ports/1/state", base, dev);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    ssize_t n = read(fd, buf, sizeof buf - 1);
    close(fd);
    if (n <= 0) return 0;
    buf[n] = '\0';
    return strstr(buf, "ACTIVE") != NULL;
}

static int add_dev(const char *name) {
    if (ndev >= MAXDEV) return -1;
    char path[512];
    snprintf(path, sizeof path, "%s/%s/ports/1/counters/port_xmit_data", base, name);
    int tx = open(path, O_RDONLY);
    snprintf(path, sizeof path, "%s/%s/ports/1/counters/port_rcv_data", base, name);
    int rx = open(path, O_RDONLY);
    if (tx < 0 || rx < 0) {
        fprintf(stderr, "cannot open counters: %s\n", name);
        if (tx >= 0) close(tx);
        if (rx >= 0) close(rx);
        return -1;
    }
    snprintf(devs[ndev].name, sizeof devs[ndev].name, "%s", name);
    devs[ndev].fd_tx = tx;
    devs[ndev].fd_rx = rx;
    ndev++;
    return 0;
}

static int cmp_str(const void *a, const void *b) {
    return strcmp(*(char *const *)a, *(char *const *)b);
}

static void discover_active(void) {
    DIR *d = opendir(base);
    if (!d) {
        perror(base);
        return;
    }
    char *names[MAXDEV * 4];
    int n = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL && n < MAXDEV * 4) {
        if (e->d_name[0] == '.') continue;
        if (port_active(e->d_name)) names[n++] = strdup(e->d_name);
    }
    closedir(d);
    qsort(names, n, sizeof names[0], cmp_str);
    for (int i = 0; i < n; i++) {
        add_dev(names[i]);
        free(names[i]);
    }
}

static void add_list(const char *list) {
    char *copy = strdup(list), *save = NULL;
    for (char *tok = strtok_r(copy, ",", &save); tok; tok = strtok_r(NULL, ",", &save))
        add_dev(tok);
    free(copy);
}

int main(int argc, char **argv) {
    long long interval_ns = 1000000LL;
    long iters = 0;
    int core = -1, realtime = 0;
    const char *outfile = NULL, *devlist = NULL;

    int opt;
    while ((opt = getopt(argc, argv, "i:n:d:b:c:ro:h")) != -1) {
        switch (opt) {
        case 'i': interval_ns = (long long)(atof(optarg) * 1000.0); break;
        case 'n': iters = atol(optarg); break;
        case 'd': devlist = optarg; break;
        case 'b': base = optarg; break;
        case 'c': core = atoi(optarg); break;
        case 'r': realtime = 1; break;
        case 'o': outfile = optarg; break;
        default:
            fprintf(stderr, "usage: %s [-i us] [-n ticks] [-d dev,..] [-b sysfs_root] [-c core] [-r] [-o file]\n",
                    argv[0]);
            return 2;
        }
    }
    if (interval_ns < 1000) interval_ns = 1000;

    /* When the reader (the monitor service) exits, the next write gets SIGPIPE/EPIPE and the
     * process ends. PR_SET_PDEATHSIG is not used: that signal fires when the starting THREAD
     * exits, and the service may start the sampler from a short-lived thread. */

    if (devlist) add_list(devlist); else discover_active();
    if (ndev == 0) {
        fprintf(stderr, "no active RDMA device found under %s\n", base);
        return 1;
    }

    if (outfile && !freopen(outfile, "a", stdout)) {
        perror(outfile);
        return 1;
    }
    setvbuf(stdout, NULL, _IOFBF, 1 << 16);

    if (core >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(core, &set);
        sched_setaffinity(0, sizeof set, &set);
    }
    if (realtime) {
        struct sched_param sp = { .sched_priority = 10 };
        if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0)
            fprintf(stderr, "SCHED_FIFO not permitted (no CAP_SYS_NICE); running at normal priority\n");
    }

    printf("H,2,%lld,%d\n", interval_ns / 1000, ndev);
    for (int i = 0; i < ndev; i++) printf("D,%d,%s\n", i, devs[i].name);
    long long anchor_at = clock_ns(CLOCK_MONOTONIC_RAW);
    printf("C,%lld,%lld\n", anchor_at, clock_ns(CLOCK_REALTIME));
    fflush(stdout);

    struct timespec next;
    clock_gettime(CLOCK_MONOTONIC, &next);
    long long next_ns = ts_ns(&next);

    for (long tick = 0; iters == 0 || tick < iters; tick++) {
        for (int i = 0; i < ndev; i++) {
            long long tx = 0, rx = 0;
            long long ta = clock_ns(CLOCK_MONOTONIC_RAW);
            int etx = read_cnt(devs[i].fd_tx, &tx);
            long long tb = clock_ns(CLOCK_MONOTONIC_RAW);
            long long ra = clock_ns(CLOCK_MONOTONIC_RAW);
            int erx = read_cnt(devs[i].fd_rx, &rx);
            long long rb = clock_ns(CLOCK_MONOTONIC_RAW);
            if (etx) printf("E,%ld,%d,tx,%d\n", tick, i, etx);
            if (erx) printf("E,%ld,%d,rx,%d\n", tick, i, erx);
            if (!etx && !erx)
                printf("S,%ld,%d,%lld,%lld,%lld,%lld,%lld,%lld\n", tick, i, ta, tb, tx, ra, rb, rx);
        }
        if (fflush(stdout) != 0 || ferror(stdout)) break;

        long long now_raw = clock_ns(CLOCK_MONOTONIC_RAW);
        if (now_raw - anchor_at > 10000000000LL) {
            anchor_at = now_raw;
            printf("C,%lld,%lld\n", anchor_at, clock_ns(CLOCK_REALTIME));
        }

        /* Next tick. On an overrun, start at once and re-anchor the schedule to now: no
         * backlog and no idle wait (real period = max(interval, read time)). */
        long long now = clock_ns(CLOCK_MONOTONIC);
        next_ns += interval_ns;
        if (next_ns < now) next_ns = now;
        next.tv_sec = next_ns / 1000000000LL;
        next.tv_nsec = next_ns % 1000000000LL;
        while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL) == EINTR) {}
    }
    fflush(stdout);
    return 0;
}
