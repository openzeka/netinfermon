# netinfermon

**net · infer · mon — LLM inference benchmark with RDMA traffic monitoring, for ConnectX (mlx5) hosts.**

netinfermon runs LLM benchmarks against an OpenAI-compatible endpoint (vLLM, SGLang, …) and, at the
same time, samples the RDMA traffic on the host's ConnectX NIC every ~1 ms. Both happen in one
process on one clock, so every request lines up exactly with the network traffic it caused:
prefill bursts, decode traffic, stalls, and how much of time-to-first-token is spent moving data.

Typical questions it answers:

- How much does TTFT change if the link drops from 200G to 100G? (Run the same plan on both, compare.)
- How many bytes does a model send per prompt token at TP=N, and how close do bursts get to line rate?
- Is a slow round caused by the network, by compute, or by a node that stalls everyone?

## What runs where

| Machine | What | How |
|---|---|---|
| **Node**: the host that serves the model endpoint (for multi-node tensor parallelism, the head node; in a ring all-reduce every node carries about the same traffic) | `netinfermon serve`: sampler, benchmark runner, web page, local control API. Raw data in `~/.local/share/netinfermon/` | Python ≥ 3.11 and `gcc`. No root: the RDMA counters in `/sys/class/infiniband` are world-readable |
| **Your browser** | Start runs, watch traffic and requests live, read and download reports, compare runs | `http://<node>:8502` |
| **Any other computer** (optional) | Heavy charts, comparing runs from several nodes | `netinfermon fetch` + `netinfermon report` |

Ordinary tools (`nload`, `ip -s link`, switch counters at 1 s) do not show RDMA traffic or
millisecond bursts. netinfermon reads the NIC's RDMA byte counters (`port_xmit_data`,
`port_rcv_data`) with a small C sampler and timestamps every read.

## Quickstart

On the node:

```bash
uv tool install git+https://github.com/openzeka/netinfermon.git     # or: pipx install git+https://…
netinfermon doctor          # finds RDMA devices, builds the sampler, checks counters, sample rate and ports
netinfermon serve           # → open http://<node>:8502
```

`uv tool install` puts the command in `~/.local/bin`, which is on the PATH of a normal login
shell (run `uv tool update-shell` once if it is not). On start-up Streamlit also prints an
"External URL" with your network's public address; that does not mean the page is reachable
from the internet (see [Security](#security)).

Then, in the browser:

1. **Benchmark** tab: endpoint and model, a built-in plan (e.g. `linkspeed`) or a custom workload
   (input tokens, output tokens, concurrency, rounds), a campaign name and a run label such as `200G`.
   **Start run.** The **Live** tab shows the traffic with every request's prefill (green) and
   generation (blue) on the timeline.
2. Change what you want to measure (link, settings, model build) and start the same plan again
   with another label, e.g. `100G`.
3. **Results** tab → **Build report** for each run; **Compare** tab → base `200G`, compared `100G`.

The same from the command line:

```bash
netinfermon run linkspeed-study 200G --plan linkspeed --endpoint http://127.0.0.1:8000/v1 --expect all --wait
netinfermon run linkspeed-study 100G --plan linkspeed --endpoint http://127.0.0.1:8000/v1 --expect one --wait
netinfermon report linkspeed-study/200G --charts html
netinfermon report linkspeed-study/100G --charts html
netinfermon compare linkspeed-study/200G linkspeed-study/100G
```

## Benchmarks

A **plan** is a list of points (concurrency, input tokens, output tokens) with warm-up and measured
rounds. A round sends `concurrency` requests at the same moment. Built-in plans (`netinfermon plans`):

| Plan | Points |
|---|---|
| `smoke` | one short point, ~2 minutes; check a new endpoint first |
| `linkspeed` | 1K / 8K / 32K input at concurrency 1, 8K at concurrency 4; 1024 output tokens |
| `linkspeed-ext` | `linkspeed` + 64K input at concurrency 1 and 16K at concurrency 4 |
| `concurrency` | 1K → 256 at concurrency 1, 2, 4, 8, 16 |

Your own plan is a JSON file in the same format (`netinfermon run … --plan my-plan.json`).

How requests are made, so results are comparable between runs:

- **Exact lengths.** Prompts are padded to the input length, sized on the server's own tokenizer, and
  output length is forced (`ignore_eos` + `min_tokens`, vLLM and SGLang). Every request is checked.
- **No prefix-cache hits.** Every request starts with a tag unique to the run and request, and the
  padding order is drawn per request.
- **Same prompts in every run of a campaign.** Prompt choice and padding are seeded by
  (plan seed, point, round).
- **No silent retries.** A failed or stalled request is an error, not a slow sample.
- **Resumable.** Every round is written to disk when it ends. Starting a label again continues
  where it stopped; finished rounds are skipped.
- **Link check** (`--expect`). After the first round with traffic: `all` = every RDMA device
  carries traffic, `one` = one device carries ≥ 99% (useful when emulating a slower link by limiting
  NCCL to one device), `any` = no check.

## Reports

`netinfermon report CAMPAIGN/LABEL` (or **Build report**) writes `report/` in the run directory:

| Per point (means over measured rounds) | Meaning |
|---|---|
| TTFT, decode TPS, end-to-end TPS, latency | decode TPS excludes TTFT: (output tokens − 1) ÷ (last token − first token) |
| prefill MB per input token | bytes this node sent during prefill ÷ prompt tokens |
| prefill 1 ms p50 / p90 / p99 | per-sample rate during prefill; shows collective bursts that 5 ms windows flatten |
| lead-in / transfer / gaps / stalls / tail | prefill time split in 5 ms bins: quiet before the first transfer (request handling, scheduling) / link carrying data (TX ≥ 5 Gb/s) / quiet runs < 100 ms between transfers (compute between collectives) / quiet runs ≥ 100 ms between transfers (every rank waiting, often one slow node) / quiet after the last transfer until the first token |
| 5 ms peak (≥) | guaranteed minimum of the highest 5 ms window |
| link busy | (bytes ÷ link maximum) ÷ prefill time |

`--charts html` adds one zoomable chart per point: every sample, 100 ms windows, and each round's
prefill and generation marked. `--charts html+svg` also writes SVGs (needs `matplotlib`; large for
long points).

`netinfermon compare BASE TEST` lines up two runs point by point: TTFT and decode change, the change in
prefill **transfer** time (the part a link change explains), and on both sides the stall time and
the lead-in + tail time (waiting and request handling that can differ between runs for other
reasons).

## Recording without the built-in benchmark

- Web page sidebar: **Start recording**, add markers, **Stop**; **Results → Manual recordings** gives
  a summary per marker.
- `netinfermon record CAMPAIGN LABEL -- <any command>` records while the command runs.
- The control API on `127.0.0.1:8599` (`/record/start`, `/marker`, `/record/stop`, `/status`) lets
  other tools mark their own phases.
- `netinfermon summarize recording.csv` prints the per-marker summary.

## Configuration

`~/.config/netinfermon/config.toml` (all optional):

```toml
endpoint = "http://127.0.0.1:8000/v1"
web_port = 8502
api_port = 8599
interval_us = 1000        # sampling interval
devices = []              # RDMA devices to read; empty = every ACTIVE one
insecure_tls = false      # true: accept self-signed certificates on https endpoints

[link]
max_gbps = 195            # measured maximum, all devices, per direction
one_device_gbps = 109     # measured maximum with traffic on one device only
```

Measure the link maximum once per cluster, e.g. with `ib_write_bw` (perftest) between two nodes,
with and without limiting to one device. Without it, the port's nominal rate is used.

## Security

- **The web page has no login.** `netinfermon serve` listens on `0.0.0.0:8502` by default, so anyone
  who can reach that port can watch traffic, start benchmark runs against any URL they enter, and
  download reports. Run it only on a trusted network, or bind it to the node itself and use an SSH
  tunnel: `netinfermon serve --host 127.0.0.1`, then `ssh -L 8502:127.0.0.1:8502 <node>`.
- The control API (`/record/*`, `/run/*`) listens on `127.0.0.1` only. Recording paths must be
  under `$HOME`.
- Certificates on `https://` endpoints are verified. For an endpoint with a self-signed
  certificate, set `insecure_tls = true` in the config (or `NETINFERMON_INSECURE_TLS=1`).
- The sampler only reads `/sys/class/infiniband` counters and needs no root or capabilities.

## Notes for multi-host NICs (e.g. DGX Spark)

One physical port can appear as several RDMA devices (DGX Spark: `rocep1s0f1` and
`roceP2p1s0f1`, one per PCIe link). netinfermon sums all active devices, so values are for the whole
port. Limiting NCCL to one of them (`NCCL_IB_HCA`) emulates a link with roughly half the bandwidth;
use `--expect one` for such runs and `--expect all` for the normal ones.

## Development

```bash
python3 -m unittest discover -s tests -v      # fake sysfs, stand-in LLM server; no NIC or GPU needed
```

| Module | Role |
|---|---|
| `sampler/rdma_sampler.c` | reads the counters every tick; each read bracketed by CLOCK_MONOTONIC_RAW timestamps |
| `rates.py` | exact window rates with hard bounds, live state, recording summaries |
| `monitor.py` | the service: sampler process, recordings, runs, control API |
| `loadgen.py` | streaming client, prompt padding and sizing, plan runner |
| `analysis.py`, `charts.py`, `compare.py` | reports, charts, comparisons |
| `web/app.py`, `cli.py` | web page, command line |

## Credits

The request code (streaming client, forced output length, input sizing on the server's tokenizer,
prompt padding, the output-length hint) and the prompt set (`netinfermon/data/prompts.txt`) are
adapted from [openzeka/llm-benchmark](https://github.com/openzeka/llm-benchmark), licensed under GPL-3.0.

## License

netinfermon is licensed under the [GNU General Public License v3.0](LICENSE).
