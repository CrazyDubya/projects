# Many Small Workers vs Threaded Workers — Bench Experiment

## Hypothesis

The "lots of small workers, each consuming one CPU time-slice" pattern (Erlang
BEAM, Go goroutines, Java virtual threads, async coroutines) buys you
**IO-concurrency-without-thread-cost**, not raw CPU efficiency. On a 2-vCPU
box, the win is invisible when work is fast and pure; it becomes legible the
moment IO latency dominates and concurrent request count exceeds the OS-thread
ceiling.

This experiment grounds-truths that claim with a side-by-side bench. ~$5 of
VPS time, ~1 hour of setup, no production change.

## What we're testing

Two equivalent web services, hitting the same Postgres, each exposing one
endpoint that returns 50 contacts via `pg_trgm` fuzzy match:

| Stack | Worker model | Concurrency primitive |
|---|---|---|
| **Sync** | Flask + Gunicorn `gthread` (5 workers × 2 threads = 10 OS threads) | One OS thread per in-flight request |
| **Async** | Quart + Uvicorn + asyncpg + httpx async | Coroutines on a single event loop per worker |

Both stacks hit the same Postgres, same query, same hardware. The only
variable is the concurrency model.

## Setup

### Hardware target
- 2 vCPUs, 4-8 GiB RAM
- Match Bluebird's prod sizing if you want directly comparable numbers
- Hetzner CPX11, DigitalOcean s-2vcpu-2gb, or Linode 4GB all work

### One-time install on the VPS
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv postgresql postgresql-contrib wrk
sudo systemctl start postgresql
sudo -u postgres psql -c "CREATE DATABASE bench;"
sudo -u postgres psql -d bench -c "CREATE EXTENSION pg_trgm;"

# Seed: 100k synthetic contacts
sudo -u postgres psql -d bench <<'SQL'
CREATE TABLE contacts (
  id SERIAL PRIMARY KEY,
  first_name VARCHAR(80),
  last_name VARCHAR(80),
  city VARCHAR(80)
);
INSERT INTO contacts (first_name, last_name, city)
SELECT
  (ARRAY['John','Mary','Bob','Alice','Carlos','Maria','Linda','David','Sarah','Michael'])[1 + (random()*9)::int],
  (ARRAY['Smith','Jones','Garcia','Lee','Patel','Cohen','Nguyen','Brown','Davis','Wilson'])[1 + (random()*9)::int],
  (ARRAY['Staten Island','Brooklyn','Queens','Manhattan','Bronx','Jersey City','Hoboken','Newark'])[1 + (random()*7)::int]
FROM generate_series(1, 100000);
CREATE INDEX contacts_lastname_trgm ON contacts USING GIN (last_name gin_trgm_ops);
ANALYZE contacts;
SQL

python3 -m venv .venv
source .venv/bin/activate
pip install flask gunicorn quart uvicorn psycopg[binary] asyncpg httpx
```

## The two services

Both expose `GET /search?q=<lastname>` and return 50 matched contacts.

### `sync_app.py` — Flask + gthread
```python
from flask import Flask, jsonify, request
import psycopg

app = Flask(__name__)
DSN = "postgresql:///bench"

@app.route("/search")
def search():
    q = request.args.get("q", "Smith")
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, first_name, last_name, city,
                   similarity(last_name, %s) AS sim
            FROM contacts
            WHERE last_name %% %s
            ORDER BY sim DESC
            LIMIT 50
        """, (q, q))
        rows = [
            {"id": r[0], "first_name": r[1], "last_name": r[2],
             "city": r[3], "sim": float(r[4])}
            for r in cur.fetchall()
        ]
    return jsonify(rows)
```

### `async_app.py` — Quart + asyncpg
```python
from quart import Quart, request, jsonify
import asyncpg

app = Quart(__name__)
DSN = "postgresql:///bench"
_pool = None

@app.before_serving
async def _open_pool():
    global _pool
    _pool = await asyncpg.create_pool(DSN, min_size=4, max_size=20)

@app.route("/search")
async def search():
    q = request.args.get("q", "Smith")
    async with _pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, first_name, last_name, city,
                   similarity(last_name, $1) AS sim
            FROM contacts
            WHERE last_name % $1
            ORDER BY sim DESC
            LIMIT 50
        """, q)
    return jsonify([dict(r) for r in rows])
```

### Run them
```bash
# Terminal A — sync on :5001
gunicorn -w 5 --threads 2 -b 0.0.0.0:5001 sync_app:app

# Terminal B — async on :5002
uvicorn async_app:app --host 0.0.0.0 --port 5002 --workers 2
```

## Bench protocol

Two bench passes, comparing identical conditions:

### Pass 1 — no artificial IO latency (~5ms query time)
```bash
wrk -t4 -c200 -d30s "http://localhost:5001/search?q=Smith"   # sync
wrk -t4 -c200 -d30s "http://localhost:5002/search?q=Smith"   # async
```

### Pass 2 — simulated downstream latency (100ms `pg_sleep`)
Patch both apps to add `pg_sleep(0.1)` to the query (anywhere — it just has to
add 100ms of IO wait per request), then re-bench:
```bash
# In sync_app.py:
# cur.execute("SELECT pg_sleep(0.1), ...")
# In async_app.py:
# await conn.fetch("SELECT pg_sleep(0.1), ...")
wrk -t4 -c200 -d30s "http://localhost:5001/search?q=Smith"
wrk -t4 -c200 -d30s "http://localhost:5002/search?q=Smith"
```

## What you should see

### Pass 1 (CPU/DB-bound, no IO wait)
Roughly identical throughput. Maybe sync wins slightly because no event-loop
overhead. Both pin the 2 vCPUs.

| Metric | Sync (gthread) | Async (Quart) |
|---|---|---|
| Req/s | ~800-1200 | ~700-1100 |
| p50 latency | 20-30ms | 25-35ms |
| p99 latency | 80-150ms | 80-180ms |

**Interpretation:** with no waiting, the pattern doesn't help. Each request is
~5ms of work; gthread serves it on a thread, Quart on a coroutine. Same
result. The "lots of small workers" trick has nothing to swap to.

### Pass 2 (IO-bound, 100ms per request)
Sync plateaus at ~100 req/s (10 slots × 10 req/s each). Async pulls away
hard.

| Metric | Sync (gthread) | Async (Quart) |
|---|---|---|
| Req/s | ~100 | ~1500-1800 |
| p50 latency | 800-1500ms | 110-130ms |
| p99 latency | 2-5s (queueing) | 150-250ms |

**Interpretation:** sync is bounded by `(workers × threads) / per-request-time
= 10 / 0.1s = 100 req/s`. Every additional concurrent request queues behind a
busy thread. Async runs all 200 connections on 2 event loops; while one waits
on Postgres, others get scheduled. The 2 vCPUs go from "mostly idle" (sync)
to "still mostly idle, but serving 15× more requests" (async).

## How to read the result

The exact numbers will vary by VPS provider and noise level. The shape is
what matters:

- **Pass 1 ratio close to 1:1** → confirms the pattern doesn't add CPU
  efficiency on its own.
- **Pass 2 ratio of ~10-20×** → confirms the pattern's actual win is IO
  concurrency under load that exceeds your OS-thread budget.

If you see Pass 1 async significantly slower than sync, the event-loop
overhead per request is too high — likely a misconfigured asyncpg pool or
running uvicorn with `--workers 1`.

## Why this matters for Bluebird

Bluebird today runs at **~10-50 simultaneous CRM requests at peak**, with
gthread giving 10 slots. Pass 2's regime (200+ concurrent users, each waiting
on slow downstream IO) is **~10× current peak**. Until usage grows that much,
or until a single endpoint starts blocking for 100ms+ per request, the win
this experiment demonstrates won't move Bluebird's needle.

If usage grows or downstream APIs slow down: this bench tells you exactly
which endpoints to rewrite first (the slow ones), and what the speedup will
be.

## Cost and cleanup

- VPS: ~$0.01/hour for the cheapest 2-vCPU machine; whole experiment runs in
  under an hour.
- Cleanup: `sudo systemctl stop postgresql && rm -rf .venv && exit` → destroy
  the VPS from the provider console.

## Variants worth running if you have extra time

1. **Add `wrk -c500` and `-c1000`**: watch sync collapse harder while async
   keeps serving. Asyncpg's pool size becomes the new bottleneck.
2. **Swap async for Erlang/Elixir Phoenix**: same endpoint, same query.
   Compare BEAM's reduction-counter scheduling against asyncio's cooperative
   model. BEAM should match async on throughput but with tighter p99 tails.
3. **Run with PYTHON 3.14 no-GIL build**: rebench Pass 1 with multiple CPU-
   bound workers (e.g. add a Python regex loop to each request). You'll see
   no-GIL pull ahead of GIL Python on CPU-bound paths only.
