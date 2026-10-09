# MedBridge Load Test

- **Generated:** 2026-10-09T16:16:56.825886+00:00
- **Base URL:** `http://127.0.0.1:8100`
- **Requests per user:** 1; query mix: emergency, small_talk, general, out_of_domain, personal, clarification
- **Server rate limit needed:** RATE_LIMIT_PER_MINUTE >= 2 (or 0)

> A single local Ollama server serialises LLM calls (one at a time), so concurrent requests queue and per-request latency grows with the number of users; the emergency fast-path and small-talk paths make no LLM call and are not affected.

> The API rate limiter counts per client IP. All simulated users share one IP, so the server under test needs RATE_LIMIT_PER_MINUTE at least as large as the number of message requests sent within one minute (or 0 to disable). Session creation is limited to 10/min per IP; the load test waits for Retry-After and does not count that wait as latency.

| Users | Requests | OK | Errors | 429 | Error rate | p50 ms | p95 ms | max ms | Throughput (req/s) | Wall s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1 | 0 | 0 | 0.0% | 9.0 | 9.0 | 9.0 | 109.883 | 0.01 |
| 2 | 2 | 2 | 0 | 0 | 0.0% | 7.9 | 8.9 | 8.9 | 220.025 | 0.01 |

## p50 latency by query kind (ms)

| Users | emergency | small_talk |
| ---: | ---: | ---: |
| 1 | 9.0 | None |
| 2 | 7.9 | 8.9 |
