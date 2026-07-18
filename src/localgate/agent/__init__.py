"""``localgate code`` — a minimal agentic coding loop over a chat-completions backend.

This is deliberately small: three filesystem tools (`agent.tools`) and a loop that
feeds tool calls back to the model until it answers in plain text (`agent.loop`).
See CODING_AGENT_PLAN.md for what this is a first step toward.
"""

from __future__ import annotations
