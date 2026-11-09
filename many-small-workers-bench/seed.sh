#!/bin/bash
# One-time setup: create bench DB, install pg_trgm, seed 100k synthetic
# contacts with a trigram index on last_name.
#
# Usage: ./seed.sh
# Requires: a local postgres reachable as the current user (or set PGUSER etc).

set -e

DB="${BENCH_DB:-bench}"

echo "→ Creating database $DB (if not exists)"
psql -tAc "SELECT 1 FROM pg_database WHERE datname='$DB'" | grep -q 1 \
    || createdb "$DB"

echo "→ Installing pg_trgm and seeding 100k contacts"
psql -d "$DB" <<SQL
CREATE EXTENSION IF NOT EXISTS pg_trgm;

DROP TABLE IF EXISTS contacts;
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

CREATE INDEX IF NOT EXISTS contacts_lastname_trgm
    ON contacts USING GIN (last_name gin_trgm_ops);

ANALYZE contacts;

SELECT count(*) AS contacts_seeded FROM contacts;
SQL

echo "✓ Seed complete."
