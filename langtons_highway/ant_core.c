/*
 * ant_core.c - fast Langton's Ant simulator with a machine-checked
 * "highway certificate".
 *
 * Usage:
 *   ant_core N AX AY START END MAX_STEPS [SEED]
 *
 * Simulates every N x N starting pattern with index in [START, END).
 * With SEED given, simulates END-START uniformly random patterns instead
 * (xorshift64, so a run is reproducible from its seed).
 * Pattern bit b (0 = LSB) sets cell (x = b % N, y = b / N) black.
 * The ant starts on cell (AX, AY) heading north (+y).
 *
 * One CSV line per pattern on stdout:
 *   index,status,onset,cert_step,period,dx,dy,visited,bbox_w,bbox_h,onset_exact
 *
 * status:  HIGHWAY  certified periodic-with-drift forever (see README)
 *          TIMEOUT  no certificate within MAX_STEPS
 *          OOB      ant left the +/- (G/2) grid
 *
 * The certificate (proved in README.md) at step s with period P and
 * drift d = (dx,dy) != 0:
 *   (A) steps [s-P, s) are an exact translate by d of steps [s-2P, s-P):
 *       same colours read, same headings, positions offset by d.
 *   (B) for every step u in [s-P, s) reading cell c:
 *       (B1) c had never been visited before u, was white, and every cell
 *            c + k*d (k >= 1) that lies inside the bounding box of all
 *            cells visited so far is unvisited and white;   or
 *       (B2) the last write to c before u happened at a step >= s-2P.
 * If (A) and (B) hold, the ant repeats the period forever, translating
 * by d each time (the highway).
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define G 4096            /* grid side; origin at centre               */
#define HALF (G / 2)
#define RB 16384          /* ring buffer length (power of two)         */
#define RBM (RB - 1)
#define MAXP 2048         /* longest period the generic search tries   */

static uint8_t *color;    /* 0 white, 1 black                           */
static int32_t *lastw;    /* step of last write, -1 never               */
static int32_t *firstv;   /* step of first visit, -1 never              */
static int32_t *touched;  /* cells to reset after a run                 */
static long ntouched;

/* ring buffers indexed by step & RBM */
static int32_t rpx[RB], rpy[RB];   /* ant position (cell read) at step   */
static uint8_t rread[RB];          /* colour read                        */
static uint8_t rhead[RB];          /* heading before the step            */
static int32_t rprevw[RB];         /* lastw of the cell before this step */
static uint8_t rfresh[RB];         /* cell never visited before this step*/

static const int DX[4] = {0, 1, 0, -1};
static const int DY[4] = {1, 0, -1, 0};

static inline long idx(int x, int y) { return (long)(y + HALF) * G + (x + HALF); }

typedef struct {
    const char *status;
    long onset, cert_step, period, dx, dy, visited, bw, bh;
    int onset_exact;
} Result;

/* certificate check; returns 1 on success and fills d/onset info */
static int certify(long s, long P, long bminx, long bmaxx, long bminy, long bmaxy,
                   long *out_dx, long *out_dy, long *onset, int *onset_exact)
{
    long a = s - P, b = s - 2 * P;
    if (b < 0) return 0;
    long dx = rpx[a & RBM] - rpx[b & RBM];
    long dy = rpy[a & RBM] - rpy[b & RBM];
    if (dx == 0 && dy == 0) return 0;
    if (rhead[a & RBM] != rhead[b & RBM]) return 0;
    /* (A) */
    for (long i = 0; i < P; i++) {
        long u = (a + i) & RBM, v = (b + i) & RBM;
        if (rread[u] != rread[v]) return 0;
        if (rpx[u] - rpx[v] != dx || rpy[u] - rpy[v] != dy) return 0;
    }
    /* (B) */
    for (long u = a; u < s; u++) {
        long r = u & RBM;
        if (rfresh[r]) {
            if (rread[r] != 0) return 0;
            long qx = rpx[r], qy = rpy[r];
            for (long k = 1;; k++) {
                qx += dx; qy += dy;
                if (qx < bminx || qx > bmaxx || qy < bminy || qy > bmaxy) break;
                long q = idx((int)qx, (int)qy);
                if (firstv[q] != -1 || color[q] != 0) return 0;
            }
        } else {
            if (rprevw[r] < b) return 0;
        }
    }
    /* onset: walk back while the translate relation keeps holding */
    long t = a;                     /* steps [t, s-P) match [t+P, s) */
    long limit = s - RB + 1;        /* oldest step still in the buffer */
    if (limit < 0) limit = 0;
    *onset_exact = 1;
    while (t > limit) {
        long u = (t - 1) & RBM, v = (t - 1 + P) & RBM;
        if (rread[u] != rread[v] || rhead[u] != rhead[v] ||
            rpx[v] - rpx[u] != dx || rpy[v] - rpy[u] != dy) break;
        t--;
    }
    if (t == limit && limit > 0) *onset_exact = 0;
    *onset = t;
    *out_dx = dx; *out_dy = dy;
    return 1;
}

static Result run(int n, int ax, int ay, uint64_t pattern, long max_steps)
{
    Result R = {"TIMEOUT", -1, -1, -1, 0, 0, 0, 0, 0, 0};
    ntouched = 0;
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++)
            if ((pattern >> (y * n + x)) & 1) {
                long q = idx(x, y);
                color[q] = 1;
                touched[ntouched++] = (int32_t)q;
            }
    int x = ax, y = ay, h = 0;
    long bminx = 0, bmaxx = n - 1, bminy = 0, bmaxy = n - 1; /* includes pattern box */
    long visited = 0;
    for (long s = 0; s < max_steps; s++) {
        long q = idx(x, y);
        uint8_t c = color[q];
        long r = s & RBM;
        rpx[r] = x; rpy[r] = y; rread[r] = c; rhead[r] = (uint8_t)h;
        rprevw[r] = lastw[q];
        rfresh[r] = (firstv[q] == -1);
        if (firstv[q] == -1) {
            firstv[q] = (int32_t)s;
            visited++;
            touched[ntouched++] = (int32_t)q;
            if (x < bminx) bminx = x;
            if (x > bmaxx) bmaxx = x;
            if (y < bminy) bminy = y;
            if (y > bmaxy) bmaxy = y;
        }
        lastw[q] = (int32_t)s;
        color[q] = c ^ 1;
        h = c ? (h + 3) & 3 : (h + 1) & 3;
        x += DX[h]; y += DY[h];
        if (x <= -HALF || x >= HALF - 1 || y <= -HALF || y >= HALF - 1) {
            R.status = "OOB"; R.cert_step = s; break;
        }
        long t = s + 1;             /* number of steps completed */
        /* cheap check for the classic 104-step highway */
        int hit = 0; long P = 0;
        if (t >= 3 * 104) {
            long b = (t - 104) & RBM, c2 = (t - 208) & RBM, d2 = (t - 312) & RBM;
            /* position at step t is where the ant stands now (x,y) */
            long ddx = x - rpx[b], ddy = y - rpy[b];
            if ((ddx || ddy) && rpx[b] - rpx[c2] == ddx && rpy[b] - rpy[c2] == ddy &&
                rpx[c2] - rpx[d2] == ddx && rpy[c2] - rpy[d2] == ddy) { hit = 1; P = 104; }
        }
        if (!hit && (t & 4095) == 0 && t >= 3 * MAXP) {
            for (long p = 1; p <= MAXP; p++) {
                long b = (t - p) & RBM, c2 = (t - 2 * p) & RBM, d2 = (t - 3 * p) & RBM;
                long ddx = x - rpx[b], ddy = y - rpy[b];
                if ((ddx || ddy) && rpx[b] - rpx[c2] == ddx && rpy[b] - rpy[c2] == ddy &&
                    rpx[c2] - rpx[d2] == ddx && rpy[c2] - rpy[d2] == ddy) { hit = 1; P = p; break; }
            }
        }
        if (hit) {
            /* store current position as step t's read position for (A) */
            long rr = t & RBM;
            rpx[rr] = x; rpy[rr] = y; rhead[rr] = (uint8_t)h; rread[rr] = color[idx(x, y)];
            long ddx, ddy, onset; int exact;
            if (certify(t, P, bminx, bmaxx, bminy, bmaxy, &ddx, &ddy, &onset, &exact)) {
                R.status = "HIGHWAY"; R.onset = onset; R.cert_step = t; R.period = P;
                R.dx = ddx; R.dy = ddy; R.onset_exact = exact;
                break;
            }
        }
    }
    if (R.cert_step < 0) R.cert_step = max_steps;
    R.visited = visited; R.bw = bmaxx - bminx + 1; R.bh = bmaxy - bminy + 1;
    for (long i = 0; i < ntouched; i++) {
        long q = touched[i];
        color[q] = 0; lastw[q] = -1; firstv[q] = -1;
    }
    return R;
}

int main(int argc, char **argv)
{
    if (argc < 7) {
        fprintf(stderr, "usage: %s N AX AY START END MAX_STEPS [SEED]\n", argv[0]);
        return 2;
    }
    int n = atoi(argv[1]), ax = atoi(argv[2]), ay = atoi(argv[3]);
    uint64_t start = strtoull(argv[4], NULL, 10), end = strtoull(argv[5], NULL, 10);
    long max_steps = atol(argv[6]);
    uint64_t rng = argc > 7 ? strtoull(argv[7], NULL, 10) * 0x9E3779B97F4A7C15ULL + 1 : 0;
    if (n < 1 || n > 8 || max_steps < 1000) { fprintf(stderr, "bad args\n"); return 2; }
    color = calloc((size_t)G * G, 1);
    lastw = malloc((size_t)G * G * sizeof(int32_t));
    firstv = malloc((size_t)G * G * sizeof(int32_t));
    touched = malloc((size_t)(max_steps + n * n + 8) * sizeof(int32_t));
    if (!color || !lastw || !firstv || !touched) { fprintf(stderr, "oom\n"); return 1; }
    memset(lastw, 0xFF, (size_t)G * G * sizeof(int32_t));
    memset(firstv, 0xFF, (size_t)G * G * sizeof(int32_t));
    uint64_t mask = n * n == 64 ? ~0ULL : ((1ULL << (n * n)) - 1);
    for (uint64_t i = start; i < end; i++) {
        uint64_t p = i;
        if (rng) {                      /* xorshift64 */
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            p = rng & mask;
        }
        Result R = run(n, ax, ay, p, max_steps);
        printf("%llu,%s,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%d\n", (unsigned long long)p, R.status,
               R.onset, R.cert_step, R.period, R.dx, R.dy, R.visited, R.bw, R.bh, R.onset_exact);
    }
    return 0;
}
