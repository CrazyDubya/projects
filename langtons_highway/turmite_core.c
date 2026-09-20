/*
 * turmite_core.c - n-colour turmite simulator with a machine-checked
 * highway certificate.
 *
 * A turmite rule is a string of L/R, one character per colour.  On a cell of
 * colour k the ant turns rule[k], sets the cell to colour (k+1) mod n, and
 * steps forward.  "RL" is Langton's Ant.
 *
 * Usage:
 *   turmite_core RULE MAX_STEPS
 *
 * Prints one CSV line:
 *   rule,status,onset,cert_step,period,dx,dy,visited,bbox_w,bbox_h,onset_exact
 *
 * status:  HIGHWAY  certified periodic-with-drift forever
 *          CYCLE    the exact state (position, heading, whole grid) repeated,
 *                   so the trajectory is periodic forever.  Detected with
 *                   Brent's algorithm over an incremental Zobrist-style hash
 *                   of the grid; a false CYCLE needs a 64-bit hash collision.
 *          TIMEOUT  no certificate within MAX_STEPS, still growing
 *          OOB      left the grid (grows without certifying a highway)
 *
 * The certificate is the n-colour generalisation of the one in ant_core.c:
 * "white" becomes "colour 0", the background.  A cell's colour is its visit
 * count mod n, so the translate argument carries over unchanged.
 *
 * Author: Claude (with Stephen)
 * Created: 2026-09-20
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define G 2048            /* grid side; origin at centre */
#define HALF (G / 2)
#define RB 8192           /* ring buffer length (power of two) */
#define RBM (RB - 1)
#define MAXP 2048         /* longest period searched */
#define MAXC 16           /* most colours supported */

static uint8_t *color;    /* colour of each cell, 0 = background */
static uint64_t *zob;     /* per-cell random weight for the grid hash */
static int32_t *lastw;    /* step of last write, -1 never */
static int32_t *firstv;   /* step of first visit, -1 never */

static int32_t rpx[RB], rpy[RB];
static uint8_t rread[RB];
static uint8_t rhead[RB];
static int32_t rprevw[RB];
static uint8_t rfresh[RB];

static const int DX[4] = {0, 1, 0, -1};
static const int DY[4] = {1, 0, -1, 0};

static inline long idx(int x, int y) { return (long)(y + HALF) * G + (x + HALF); }

typedef struct {
    const char *status;
    long onset, cert_step, period, dx, dy, visited, bw, bh;
    int onset_exact;
} Result;

static int certify(long s, long P, long bminx, long bmaxx, long bminy, long bmaxy,
                   long *out_dx, long *out_dy, long *onset, int *onset_exact)
{
    long a = s - P, b = s - 2 * P;
    if (b < 0) return 0;
    long dx = rpx[a & RBM] - rpx[b & RBM];
    long dy = rpy[a & RBM] - rpy[b & RBM];
    if (dx == 0 && dy == 0) return 0;
    if (rhead[a & RBM] != rhead[b & RBM]) return 0;
    for (long i = 0; i < P; i++) {              /* (A) exact translate */
        long u = (a + i) & RBM, v = (b + i) & RBM;
        if (rread[u] != rread[v]) return 0;
        if (rpx[u] - rpx[v] != dx || rpy[u] - rpy[v] != dy) return 0;
    }
    for (long u = a; u < s; u++) {              /* (B) */
        long r = u & RBM;
        if (rfresh[r]) {
            if (rread[r] != 0) return 0;
            long qx = rpx[r], qy = rpy[r];
            for (;;) {
                qx += dx; qy += dy;
                if (qx < bminx || qx > bmaxx || qy < bminy || qy > bmaxy) break;
                long q = idx((int)qx, (int)qy);
                if (firstv[q] != -1 || color[q] != 0) return 0;
            }
        } else {
            if (rprevw[r] < b) return 0;
        }
    }
    long t = a;                                  /* walk the onset backwards */
    long limit = s - RB + 1;
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

static Result run(const int8_t *turn, int ncol, long max_steps)
{
    Result R = {"TIMEOUT", -1, -1, -1, 0, 0, 0, 0, 0, 0};
    memset(color, 0, (size_t)G * G);
    memset(lastw, 0xFF, (size_t)G * G * sizeof(int32_t));
    memset(firstv, 0xFF, (size_t)G * G * sizeof(int32_t));

    int x = 0, y = 0, h = 0;
    long bminx = 0, bmaxx = 0, bminy = 0, bmaxy = 0, visited = 0;
    /* Brent's cycle detection over (x, y, heading, grid hash) */
    uint64_t ghash = 0, t_hash = 0;
    int32_t t_x = 0, t_y = 0; uint8_t t_h = 0;
    long power = 1, lam = 0;
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
            if (x < bminx) bminx = x;
            if (x > bmaxx) bmaxx = x;
            if (y < bminy) bminy = y;
            if (y > bmaxy) bmaxy = y;
        }
        lastw[q] = (int32_t)s;
        uint8_t nc = (uint8_t)((c + 1) % ncol);
        ghash += zob[q] * (uint64_t)((int64_t)nc - (int64_t)c);
        color[q] = nc;
        h = (h + turn[c]) & 3;
        x += DX[h]; y += DY[h];
        if (x <= -HALF + 1 || x >= HALF - 2 || y <= -HALF + 1 || y >= HALF - 2) {
            R.status = "OOB"; R.cert_step = s; break;
        }
        if (x == t_x && y == t_y && h == t_h && ghash == t_hash && s > 0) {
            R.status = "CYCLE"; R.cert_step = s; R.period = lam; break;
        }
        if (power == lam) { t_x = x; t_y = y; t_h = (uint8_t)h; t_hash = ghash;
                            power *= 2; lam = 0; }
        lam++;
        long t = s + 1;
        if ((t & 1023) == 0 && t >= 3 * 4) {
            long best = 0;
            long pmax = t / 3; if (pmax > MAXP) pmax = MAXP;
            for (long p = 1; p <= pmax; p++) {
                long b = (t - p) & RBM, c2 = (t - 2 * p) & RBM, d2 = (t - 3 * p) & RBM;
                long ddx = x - rpx[b], ddy = y - rpy[b];
                if ((ddx || ddy) && rpx[b] - rpx[c2] == ddx && rpy[b] - rpy[c2] == ddy &&
                    rpx[c2] - rpx[d2] == ddx && rpy[c2] - rpy[d2] == ddy) { best = p; break; }
            }
            if (best) {
                long rr = t & RBM;
                rpx[rr] = x; rpy[rr] = y; rhead[rr] = (uint8_t)h;
                rread[rr] = color[idx(x, y)];
                long ddx, ddy, onset; int exact;
                if (certify(t, best, bminx, bmaxx, bminy, bmaxy, &ddx, &ddy, &onset, &exact)) {
                    R.status = "HIGHWAY"; R.onset = onset; R.cert_step = t;
                    R.period = best; R.dx = ddx; R.dy = ddy; R.onset_exact = exact;
                    break;
                }
            }
        }
    }
    if (R.cert_step < 0) R.cert_step = max_steps;
    R.visited = visited; R.bw = bmaxx - bminx + 1; R.bh = bmaxy - bminy + 1;
    return R;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s RULE MAX_STEPS\n", argv[0]); return 2; }
    const char *rule = argv[1];
    long max_steps = atol(argv[2]);
    int ncol = (int)strlen(rule);
    if (ncol < 1 || ncol > MAXC) { fprintf(stderr, "rule length 1..%d\n", MAXC); return 2; }
    int8_t turn[MAXC];
    for (int i = 0; i < ncol; i++) {
        if (rule[i] == 'R') turn[i] = 1;
        else if (rule[i] == 'L') turn[i] = 3;
        else { fprintf(stderr, "rule must be L/R only\n"); return 2; }
    }
    color = malloc((size_t)G * G);
    zob = malloc((size_t)G * G * sizeof(uint64_t));
    if (zob) {   /* splitmix64, so the weights are reproducible */
        uint64_t z = 0x9E3779B97F4A7C15ULL;
        for (size_t i = 0; i < (size_t)G * G; i++) {
            z += 0x9E3779B97F4A7C15ULL;
            uint64_t v = z;
            v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ULL;
            v = (v ^ (v >> 27)) * 0x94D049BB133111EBULL;
            zob[i] = v ^ (v >> 31);
        }
    }
    lastw = malloc((size_t)G * G * sizeof(int32_t));
    firstv = malloc((size_t)G * G * sizeof(int32_t));
    if (!color || !lastw || !firstv || !zob) { fprintf(stderr, "oom\n"); return 1; }
    Result R = run(turn, ncol, max_steps);
    printf("%s,%s,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%d\n", rule, R.status, R.onset,
           R.cert_step, R.period, R.dx, R.dy, R.visited, R.bw, R.bh, R.onset_exact);
    return 0;
}
