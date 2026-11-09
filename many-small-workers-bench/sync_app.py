"""Flask + gthread sync version of the search endpoint.

Run: gunicorn -w 5 --threads 2 -b 0.0.0.0:5001 sync_app:app
"""
import os

import psycopg
from flask import Flask, jsonify, request

app = Flask(__name__)
DSN = os.environ.get("BENCH_DSN", "postgresql:///bench")

# Toggle 100ms simulated IO wait via env var so the bench can re-run without
# editing source. Set BENCH_SLOW=1 for Pass 2.
SLOW = bool(int(os.environ.get("BENCH_SLOW", "0")))


@app.route("/search")
def search():
    q = request.args.get("q", "Smith")
    sleep_clause = "pg_sleep(0.1)," if SLOW else ""
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {sleep_clause}
                   id, first_name, last_name, city,
                   similarity(last_name, %s) AS sim
            FROM contacts
            WHERE last_name %% %s
            ORDER BY sim DESC
            LIMIT 50
            """,
            (q, q),
        )
        rows = []
        for r in cur.fetchall():
            # Skip the pg_sleep null column if present
            off = 1 if SLOW else 0
            rows.append({
                "id": r[off],
                "first_name": r[off + 1],
                "last_name": r[off + 2],
                "city": r[off + 3],
                "sim": float(r[off + 4]),
            })
    return jsonify(rows)


@app.route("/health")
def health():
    return "ok"
