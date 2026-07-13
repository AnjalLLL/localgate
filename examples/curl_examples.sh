#!/usr/bin/env bash
# Every localgate endpoint, from the shell.
#
#   export LOCALGATE_API_KEY=lg_...      # from: localgate keys create --name my-app
#   export LOCALGATE_ADMIN_KEY=...       # from: your .env
#   ./examples/curl_examples.sh
set -euo pipefail

BASE="${LOCALGATE_URL:-http://localhost:8000}"
KEY="${LOCALGATE_API_KEY:-lg_your_key_here}"
ADMIN="${LOCALGATE_ADMIN_KEY:-change-me-in-production}"
MODEL="${LOCALGATE_MODEL:-llama3}"

echo "### Is it healthy? (backend + database + warnings)"
curl -s "$BASE/health" | python3 -m json.tool

echo; echo "### Chat"
curl -s "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}"

echo; echo "### Chat, streaming (Server-Sent Events, terminated by [DONE])"
curl -sN "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count to 5\"}],\"stream\":true}"

echo; echo "### Chat with memory — teach it something..."
curl -s "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" -H "X-Session-ID: curl-demo" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"My name is Ana and I prefer Postgres.\"}]}" > /dev/null

echo "### ...then ask, sending NO history. The gateway retrieves it."
curl -s "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" -H "X-Session-ID: curl-demo" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What database do I prefer?\"}]}"

echo; echo "### The stored conversation, its rolling summary, and its memory chunks"
curl -s "$BASE/v1/conversations/curl-demo" -H "Authorization: Bearer $KEY" | python3 -m json.tool

echo; echo "### Models the gateway can serve (backend's models + your aliases)"
curl -s "$BASE/v1/models" -H "Authorization: Bearer $KEY" | python3 -m json.tool

echo; echo "### Embeddings"
curl -s "$BASE/v1/embeddings" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"input": "hello world"}' | head -c 200; echo "..."

echo; echo "### Admin: create a key (the raw key is shown exactly once)"
curl -s -X POST "$BASE/admin/keys" \
  -H "X-Admin-Key: $ADMIN" -H "Content-Type: application/json" \
  -d '{"name": "from-curl", "rate_limit_per_min": 120}' | python3 -m json.tool

echo; echo "### Admin: usage — totals, per key, per model, daily"
curl -s "$BASE/admin/usage" -H "X-Admin-Key: $ADMIN" | python3 -m json.tool

echo; echo "### Admin: export everything (no lock-in)"
curl -s "$BASE/admin/export" -H "X-Admin-Key: $ADMIN" | head -c 200; echo "..."
