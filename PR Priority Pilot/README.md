---
title: PR Priority Pilot
emoji: 🚀
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
---

# PR Priority Pilot – Code Review Prioritizer

AI agent that prioritizes pull requests for busy developers.

- 3 difficulty tasks (easy, medium, hard)
- Reward: 1.0 perfect, 0.5 off‑by‑one, 0.0 if ignores security/critical
- OpenEnv API endpoints: `/reset`, `/step`, `/state`

Set environment variables: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`.