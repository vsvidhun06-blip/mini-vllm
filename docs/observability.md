# Observability: Prometheus + Grafana

mini-vLLM exposes a Prometheus `/metrics` endpoint and ships a one-command
Prometheus + Grafana stack with a **pre-provisioned datasource and dashboard**.
Bring up the stack and the dashboard is live — no manual import, no setup
clicking.

## How to run

```bash
# 1. Start the engine on the host (serves /metrics on :8000)
uvicorn src.server.api:app --port 8000

# 2. In another shell, bring up the monitoring stack
docker-compose -f docker-compose.observability.yml up
```

Then open:

| Service | URL | Notes |
|---|---|---|
| **Grafana** | <http://localhost:3000> | Dashboard *mini-vLLM — Inference Observability* (folder **mini-vLLM**). Anonymous viewing; `admin` / `admin` to edit. |
| **Prometheus** | <http://localhost:9090> | Raw query UI; check **Status → Targets** to confirm the `mini-vllm` target is `UP`. |

Drive some traffic so the panels populate, e.g.:

```bash
for i in $(seq 1 20); do
  curl -s -X POST localhost:8000/generate \
    -H 'content-type: application/json' \
    -d '{"prompt":"The capital of France is","max_tokens":24}' >/dev/null
done
```

### How the scrape reaches the host

The engine runs on the **host**; Prometheus runs in a **container**. The scrape
target is `host.docker.internal:8000`, and the compose file maps that name to
the host gateway (`extra_hosts: host.docker.internal:host-gateway`) so it works
on Docker Desktop (Mac/Windows) and on Linux. Scrape interval is **5s**
(`observability/prometheus.yml`). If you prefer to run the engine in a container
on the same `obs` network, change the target to that service's name.

## The dashboard

Seven panels, one per signal that matters for inference serving. Screenshots:

| Panel | Screenshot |
|---|---|
| TTFT P50/P95/P99 | _![TTFT](screenshots/obs_ttft.png) — placeholder_ |
| TPOT P50/P95/P99 | _![TPOT](screenshots/obs_tpot.png) — placeholder_ |
| Queue depth | _![Queue depth](screenshots/obs_queue_depth.png) — placeholder_ |
| Cache utilisation gauge | _![Cache utilisation](screenshots/obs_cache_util.png) — placeholder_ |
| Decode batch-size histogram | _![Batch size](screenshots/obs_batch_size.png) — placeholder_ |
| Requests/sec | _![Requests/sec](screenshots/obs_rps.png) — placeholder_ |
| CUDA graph hit rate | _![Graph hit rate](screenshots/obs_graph_hit_rate.png) — placeholder_ |

*(Replace the placeholders with real captures after a run — save PNGs under
`docs/screenshots/` with the names above.)*

## What each metric means for inference quality

| Metric (series) | Type | What it tells you |
|---|---|---|
| **TTFT** — `ttft_seconds` | Histogram | **Time to first token**: admission → first token. The single-request responsiveness SLO. Climbs when a request waits behind a long prefill or behind a full batch — i.e. it's the symptom you see when **queue depth** or **cache utilisation** is high. P99 is the tail users actually complain about. |
| **TPOT** — `tpot_seconds` | Histogram | **Time per output token**: the gap between consecutive tokens during decode — streaming smoothness. On a bandwidth-bound decode it's dominated by per-step kernel-launch overhead, which is exactly what CUDA graphs collapse (see the **graph hit rate** panel). |
| **Queue depth** — `mini_vllm_queue_depth` | Gauge | Requests admitted-pending: waiting for a batch slot or for cache blocks to free. Zero in steady state; **persistently positive means you are over capacity** and TTFT is about to rise. The leading indicator. |
| **Cache utilisation** — `mini_vllm_cache_utilisation` | Gauge (%) | Fraction of the KV block pool in use. The hard ceiling on concurrency for a paged-attention engine: at ~100% the scheduler can't admit new requests (queue depth grows) and may preempt. Watch this with queue depth — they move together under load. |
| **Decode batch size** — `mini_vllm_batch_size` | Histogram | How many requests share each decode forward. Throughput per GPU-step rises with batch density, so a distribution skewed toward 1 means you're paying decode's fixed cost per request instead of amortising it — under-batched. |
| **Requests/sec** — `requests_total{status="finished"}` | Counter → rate | Completed-request throughput. The top-line capacity number; read it against queue depth to know whether you're throughput-bound (rps flat, queue growing) or simply idle. |
| **CUDA graph hit rate** — `mini_vllm_cuda_graph_hits_total{outcome}` | Counter (hit/miss) | Share of decode forwards replayed from a captured CUDA graph vs run eagerly. **0% on CPU or before any capture** — the honest eager-path baseline. When it climbs, per-step launch overhead is being eliminated and TPOT should drop in lockstep. |
| **KV evictions** — `mini_vllm_evictions_total` | Counter | Tokens dropped by H2O eviction to keep a long sequence inside the cache budget. Non-zero only when an evicting cache is in use; rising means you're trading old context for the ability to serve beyond the budget. |

The throughline: these metrics are **causally linked**, not independent dials.
Cache utilisation and queue depth are *causes*; TTFT is the *symptom*. Batch
size and graph hit rate explain *why TPOT is what it is*. The dashboard is laid
out so you can read the chain — saturation on the bottom row explaining latency
on the top row — which is the whole point of observability over a single
"it's slow" number.

## Implementation notes

- The metric instruments live in `src/server/metrics.py`; the engine has **no
  `prometheus_client` dependency**. The collector is just another `EventBus`
  subscriber, and the two engine-sourced counters (evictions, graph hits) are
  fed through optional observer callbacks (`eviction_observer`,
  `cuda_graph_observer`) the server wires in — the same pattern as the existing
  spec-decode acceptance metric.
- Queue depth and cache utilisation are derived from the per-step `pool_state`
  event (which now also carries the `waiting` count), so they update on every
  scheduler iteration with no extra plumbing.
