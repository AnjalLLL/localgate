# 2. Store embeddings as JSON arrays, not pgvector

- **Status:** Accepted
- **Date:** 2026-07-13

## Context

Memory needs to store an embedding vector per chunk and rank chunks by cosine similarity
to a query vector. The obvious answer for Postgres is `pgvector`: a native vector column,
an ANN index, and `ORDER BY embedding <=> :query LIMIT :k` pushed into SQL.

But localgate's default database is SQLite, and its stated promise is that it runs with
zero setup. SQLite has no vector type. The options were:

1. **pgvector on Postgres, something else on SQLite.** Two schemas, two query paths, two
   sets of bugs — and the SQLite path, which is the default and therefore the one most
   users are on, would be the less-tested of the two.
2. **`sqlite-vss` for SQLite.** A loadable extension. It has to be compiled or shipped per
   platform, and "zero setup" evaporates the moment a user on an unusual platform can't
   load it.
3. **JSON arrays everywhere, similarity computed in Python.**

## Decision

Store embeddings as JSON float arrays in a plain `JSON` column, and compute cosine
similarity in Python (`db/repositories/embeddings.py`).

## Rationale

The schema is then *identical* on SQLite and Postgres, there is one query path, and it is
the one every user exercises. No extension, no build step, no platform-specific packaging.

The cost is real and bounded: retrieval scans every chunk in the session and scores it in
Python. That is comfortably fine into the low thousands of chunks per session and stops
being fine well before ten thousand.

Crucially, this cost is paid **per session**, not per database. Retrieval is scoped to one
session — it has to be, or one user's conversation could surface in another's context — so
the scan size is bounded by how long a single conversation is, not by how much the gateway
has ever stored. A gateway with a million chunks across ten thousand sessions still only
scans one session's worth.

For the workload this tool is built for — a self-hosted gateway in front of a local model,
where a single inference call takes hundreds of milliseconds to seconds — a scan of a few
hundred vectors is not the bottleneck. The inference is.

## Consequences

- Zero-setup memory on SQLite. This is the property that makes the feature usable at all
  for the median user.
- Retrieval is O(chunks in session). `GET /health` warns when memory is enabled on SQLite,
  so this is visible rather than a surprise.
- **The upgrade path is deliberately preserved.** `EmbeddingRepository.search()` takes a
  query vector and returns ranked `RetrievedChunk`s. A Postgres user who outgrows the scan
  can swap the column to `vector` and the body of that one method to a pgvector query;
  nothing above the repository changes, because nothing above it knows how ranking happens.
- Vectors are excluded from `/admin/export`: they would multiply the export size while
  being reproducible from the chunk text with the same embedding model.
- `cosine_similarity` returns `0.0` for vectors of different widths rather than raising.
  Changing the embedding model changes the vector width, and old chunks written by the
  previous model should score as "no match", not take down a request.
