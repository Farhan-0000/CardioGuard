# MedBridge Load Test

- **Generated:** 2026-10-11T12:14:27.590416+00:00
- **Base URL:** `http://127.0.0.1:8000`
- **Requests per user:** 3; query mix: emergency, small_talk, general, out_of_domain, personal, clarification
- **Server rate limit needed:** RATE_LIMIT_PER_MINUTE >= 30 (or 0)

> A single local Ollama server serialises LLM calls (one at a time), so concurrent requests queue and per-request latency grows with the number of users; the emergency fast-path and small-talk paths make no LLM call and are not affected.

> The API rate limiter counts per client IP. All simulated users share one IP, so the server under test needs RATE_LIMIT_PER_MINUTE at least as large as the number of message requests sent within one minute (or 0 to disable). Session creation is limited to 10/min per IP; the load test waits for Retry-After and does not count that wait as latency.

| Users | Requests | OK | Errors | 429 | Error rate | p50 ms | p95 ms | max ms | Throughput (req/s) | Wall s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 3 | 0 | 0 | 0.0% | 28.0 | 23075.3 | 23075.3 | 0.13 | 23.13 |
| 5 | 15 | 15 | 0 | 0 | 0.0% | 56248.4 | 83853.2 | 83853.2 | 0.077 | 194.64 |
| 10 | 30 | 30 | 0 | 0 | 0.0% | 88343.8 | 162026.5 | 168321.3 | 0.09 | 331.88 |

## p50 latency by query kind (ms)

| Users | emergency | small_talk | general | out_of_domain | personal | clarification |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 28.0 | 24.5 | 23075.3 | None | None | None |
| 5 | 13.9 | 15.3 | 69746.1 | 46649.5 | 79337.1 | 79448.5 |
| 10 | 17.0 | 21.3 | 150463.5 | 84162.9 | 133528.4 | 87193.8 |
