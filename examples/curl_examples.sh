#!/usr/bin/env bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer lg_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3", "messages": [{"role": "user", "content": "Hello!"}]}'
