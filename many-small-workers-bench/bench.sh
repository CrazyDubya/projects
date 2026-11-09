#!/bin/bash
# Run the full bench: 2 passes (fast / slow IO), 2 stacks (sync / async).
# Assumes both services are already running on :5001 (sync) and :5002 (async).
#
# Usage: ./bench.sh
# Requires: wrk installed (apt-get install wrk).

set -e

DUR="${BENCH_DUR:-30s}"
CONN="${BENCH_CONN:-200}"
THREADS="${BENCH_THREADS:-4}"

bench() {
    local label="$1" url="$2"
    echo "─── $label ───"
    wrk -t"$THREADS" -c"$CONN" -d"$DUR" --latency "$url" | \
        grep -E "Requests/sec|Latency|50%|75%|90%|99%" | head -6
    echo
}

echo "════════════════════════════════════════════════════════════"
echo " PASS 1 — no artificial IO latency"
echo " (BENCH_SLOW=0 should be set in both server environments)"
echo "════════════════════════════════════════════════════════════"
bench "sync  (Flask + gthread)"    "http://localhost:5001/search?q=Smith"
bench "async (Quart + asyncpg)"   "http://localhost:5002/search?q=Smith"

echo "════════════════════════════════════════════════════════════"
echo " PASS 2 — restart both servers with BENCH_SLOW=1 first"
echo "════════════════════════════════════════════════════════════"
echo "Stop servers, re-export BENCH_SLOW=1, restart, then re-run:"
echo "  bench \"sync  (Flask + gthread, +100ms IO)\"  http://localhost:5001/search?q=Smith"
echo "  bench \"async (Quart + asyncpg, +100ms IO)\" http://localhost:5002/search?q=Smith"
echo
echo "Or run this script again after both servers are restarted with BENCH_SLOW=1."
