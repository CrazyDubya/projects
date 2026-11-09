"""Quart + asyncpg async version of the search endpoint.

Run: uvicorn async_app:app --host 0.0.0.0 --port 5002 --workers 2
"""
import os

import asyncpg
from quart import Quart, jsonify, request

app = Quart(__name__)
DSN = os.environ.get("BENCH_DSN", "postgresql:///bench")
SLOW = bool(int(os.environ.get("BENCH_SLOW", "0")))

_pool: asyncpg.Pool | None = None


@app.before_serving
async def _open_pool():
    global _pool
    _pool = await asyncpg.create_pool(DSN, min_size=4, max_size=20)


@app.after_serving
async def _close_pool():
    if _pool is not None:
        await _pool.close()


@app.route("/search")
async def search():
    q = request.args.get("q", "Smith")
    sleep_clause = "pg_sleep(0.1)," if SLOW else ""
    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT {sleep_clause}
                   id, first_name, last_name, city,
                   similarity(last_name, $1) AS sim
            FROM contacts
            WHERE last_name % $1
            ORDER BY sim DESC
            LIMIT 50
            """,
            q,
        )
    out = []
    off = 1 if SLOW else 0
    keys = ["id", "first_name", "last_name", "city", "sim"]
    for r in rows:
        out.append({k: r[i + off] for i, k in enumerate(keys)})
    return jsonify(out)


@app.route("/health")
async def health():
    return "ok"
