"""analysis.py — report for a run: request metrics joined with the RDMA traffic they caused.

Every round (the requests that start together) is split by the request timestamps, which are on
the recording's clock (CLOCK_MONOTONIC_RAW, same host):

    prefill     round start -> last first token of the round
    generation  last first token -> round end

Per round: bytes sent and received (exact, all devices), bytes per input token, mean rate,
5 ms peak (guaranteed minimum), per-sample (~1 ms) p50 / p90 / p99 during prefill, link busy
f = (bytes / link maximum) / prefill time, and the prefill split into
    transfer  5 ms bins with TX >= 5 Gb/s
    gaps      quieter runs shorter than 100 ms (compute between collectives)
    stalls    quieter runs of 100 ms or more (every rank waiting; often one slow node)

Per point: means over the measured rounds (warm-up excluded); peaks are the highest.
Writes <run>/report/{points.csv, rounds.csv, report.md} and, on request, charts.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import statistics
from array import array
from bisect import bisect_left, bisect_right

from .loadgen import read_json, read_requests, request_summary
from .rates import BYTES_PER_WORD, Recording, _grid, _sliding_peak

W5 = 5_000_000
TRANSFER_GBPS = 5.0
STALL_NS = 100_000_000


def load_recordings(run_dir: str, link_gbps: float) -> list[Recording]:
    return [Recording(p, link_gbps) for p in sorted(glob.glob(os.path.join(run_dir, "network", "recording-*.csv")))]


def covering(recs: list[Recording], t0: int, t1: int) -> int | None:
    for i, rec in enumerate(recs):
        s, e = rec.span()
        if s <= t0 and t1 <= e:
            return i
    return None


def value_at(ser, t: int) -> float | None:
    """Counter value (words) at time t, linear between the two neighbouring samples."""
    i = ser.index_at(t)
    if i < 0 or i >= len(ser) - 1:
        return float(ser.v[i]) if i >= 0 and ser.t[i] == t else None
    t0, t1, v0, v1 = ser.t[i], ser.t[i + 1], ser.v[i], ser.v[i + 1]
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)


def window_bytes(rec: Recording, d: str, t0: int, t1: int) -> tuple[float, dict] | None:
    per_dev = {}
    for dev in rec.devices:
        ser = rec.state.series[(dev, d)]
        a, b = value_at(ser, t0), value_at(ser, t1)
        if a is None or b is None:
            return None
        per_dev[dev] = (b - a) * BYTES_PER_WORD
    return sum(per_dev.values()), per_dev


def tick_rates(path: str) -> dict:
    """Rate between consecutive samples (one per sampler tick, ~1 ms), all devices summed.

    For every device the counter difference is exact; the time between the two reads is known
    to +- (h_prev + h_now), so each tick has a hard lower / upper bound. Ticks where a device's
    read failed or a counter went backwards are left out."""
    out = {k: array("d") for k in ("t", "tx", "tx_lo", "tx_hi", "rx", "rx_lo", "rx_hi")}
    devices, prev, cur, cur_tick = [], {}, {}, None

    def close():
        if len(cur) != len(devices) or not devices:
            return
        sums = {"tx": [0.0, 0.0, 0.0], "rx": [0.0, 0.0, 0.0]}
        for idx, now in cur.items():
            before = prev.get(idx)
            if before is None:
                return
            for d, k in (("tx", 0), ("rx", 1)):
                (t0, v0, h0), (t1, v1, h1) = before[k], now[k]
                dt, dv, u = t1 - t0, v1 - v0, h0 + h1
                if dt <= 0 or dv < 0:
                    return
                bits = dv * 32
                sums[d][0] += bits / dt
                sums[d][1] += bits / (dt + u)
                sums[d][2] += bits / (dt - u) if dt > u else float("inf")
        out["t"].append(sum(c[0][0] for c in cur.values()) / len(cur))
        for d in ("tx", "rx"):
            out[d].append(sums[d][0])
            out[d + "_lo"].append(sums[d][1])
            out[d + "_hi"].append(sums[d][2])

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("S,"):
                p = line.split(",")
                tick, idx = int(p[1]), int(p[2])
                if tick != cur_tick:
                    close()
                    prev.update(cur)
                    cur, cur_tick = {}, tick
                a1, b1, v1, a2, b2, v2 = (int(x) for x in p[3:9])
                cur[idx] = ((a1 + (b1 - a1) // 2, v1, (b1 - a1 + 1) // 2),
                            (a2 + (b2 - a2) // 2, v2, (b2 - a2 + 1) // 2))
            elif line.startswith("D,"):
                devices.append(line.split(",")[2])
            elif line.startswith("E,"):
                prev.pop(int(line.split(",")[2]), None)     # failed read: next tick has no partner
    close()
    return out


def _pct(xs: list[float], p: float):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(-(-len(s) * p // 1)) - 1))]


def prefill_split(rec: Recording, t0: int, t1: int) -> dict:
    """Prefill time split into transfer / short gaps / stalls, from consecutive 5 ms bins of
    TX over all devices."""
    bins = _grid([rec.state.series[(dev, "tx")] for dev in rec.devices], t0, t1, W5)
    transfer = gap = stall = 0
    run = 0

    def close_run():
        nonlocal gap, stall
        if run * W5 >= STALL_NS:
            stall += run
        else:
            gap += run

    for r in bins:
        if r.gbps >= TRANSFER_GBPS:
            close_run()
            run = 0
            transfer += 1
        else:
            run += 1
    close_run()
    return {"prefill_transfer_s": transfer * W5 / 1e9, "prefill_gap_s": gap * W5 / 1e9,
            "prefill_stall_s": stall * W5 / 1e9}


def analyze(run_dir: str, rows: list[dict], plan: dict, link_gbps: float,
            charts: str | None = None) -> tuple[list[dict], dict, dict]:
    """Per-round and per-point network numbers. charts: None, "html" or "html+svg"."""
    recs = load_recordings(run_dir, link_gbps)
    rec_paths = sorted(glob.glob(os.path.join(run_dir, "network", "recording-*.csv")))
    ticks: dict[int, dict] = {}
    rounds = {}
    for r in rows:
        if r["error"] or not r.get("t_first_ns"):
            continue
        rounds.setdefault((r["point"], int(r["round"])), []).append(r)

    out = []
    for (pid, rnd), rs in sorted(rounds.items()):
        t_start = min(int(r["t0_ns"]) for r in rs)
        t_ff = max(int(r["t_first_ns"]) for r in rs)
        t_end = max(int(r["t_end_ns"]) for r in rs)
        ri = covering(recs, t_start, t_end)
        row = {"point": pid, "round": rnd, "warmup": int(rs[0]["warmup"]), "concurrency": len(rs),
               "prompt_tokens": sum(int(r["prompt_tokens"] or 0) for r in rs),
               "prefill_s": (t_ff - t_start) / 1e9, "generation_s": (t_end - t_ff) / 1e9}
        if ri is None:
            row["note"] = "not covered by a recording"
            out.append(row)
            continue
        rec = recs[ri]
        wins = [window_bytes(rec, d, a, b) for d, a, b in (("tx", t_start, t_ff), ("rx", t_start, t_ff),
                                                            ("tx", t_ff, t_end), ("rx", t_ff, t_end),
                                                            ("tx", t_start, t_end))]
        if any(w is None for w in wins):
            row["note"] = "round edge outside the recorded samples"
            out.append(row)
            continue
        (pre_tx, _), (pre_rx, _), (gen_tx, _), (gen_rx, _), (all_tx, all_dev) = wins
        series = lambda d: [rec.state.series[(dev, d)] for dev in rec.devices]
        pk_pre = _sliding_peak(series("tx"), t_start, t_ff, W5)
        pk_tx, pk_rx = _sliding_peak(series("tx"), t_start, t_end, W5), _sliding_peak(series("rx"), t_start, t_end, W5)
        if ri not in ticks:
            ticks[ri] = tick_rates(rec_paths[ri])
        tk = ticks[ri]
        i, j = bisect_left(tk["t"], t_start), bisect_right(tk["t"], t_ff)
        per_sample = [x for x in tk["tx"][i:j] if x == x and x != float("inf")]
        dur = row["prefill_s"]
        row.update({
            "prefill_tx_GB": pre_tx / 1e9, "prefill_rx_GB": pre_rx / 1e9,
            "prefill_bytes_per_token": pre_tx / row["prompt_tokens"] if row["prompt_tokens"] else None,
            "prefill_mean_gbps": pre_tx * 8 / dur / 1e9 if dur > 0 else None,
            "prefill_1ms_p50_gbps": _pct(per_sample, 0.50), "prefill_1ms_p90_gbps": _pct(per_sample, 0.90),
            "prefill_1ms_p99_gbps": _pct(per_sample, 0.99),
            "prefill_peak5_min_gbps": pk_pre.lo if pk_pre else None,
            "prefill_link_busy": (pre_tx * 8 / (link_gbps * 1e9)) / dur if dur > 0 else None,
            **prefill_split(rec, t_start, t_ff),
            "generation_tx_GB": gen_tx / 1e9, "generation_rx_GB": gen_rx / 1e9,
            "generation_mean_gbps": gen_tx * 8 / row["generation_s"] / 1e9 if row["generation_s"] > 0 else None,
            "round_tx_GB": all_tx / 1e9,
            "round_peak5_min_tx_gbps": pk_tx.lo if pk_tx else None,
            "round_peak5_min_rx_gbps": pk_rx.lo if pk_rx else None,
            **{f"tx_share_{dev}": (v / all_tx if all_tx else None) for dev, v in all_dev.items()},
        })
        out.append(row)

    points = {}
    for p in plan["points"]:
        meas = [r for r in out if r["point"] == p["id"] and r["warmup"] == 0 and "note" not in r]
        if not meas:
            continue
        avg = lambda k: (statistics.mean(v) if (v := [r[k] for r in meas if r.get(k) is not None]) else None)
        top = lambda k: max((r[k] for r in meas if r.get(k) is not None), default=None)
        points[p["id"]] = {
            **{k: avg(k) for k in ("prefill_s", "prefill_bytes_per_token", "prefill_mean_gbps",
                                   "prefill_1ms_p50_gbps", "prefill_1ms_p90_gbps", "prefill_1ms_p99_gbps",
                                   "prefill_link_busy", "prefill_transfer_s", "prefill_gap_s",
                                   "prefill_stall_s", "generation_mean_gbps", "round_tx_GB")},
            "prefill_peak5_min_gbps": top("prefill_peak5_min_gbps"),
            **{k: avg(k) for k in meas[0] if k.startswith("tx_share_")},
        }

    ceiling = {d: max((r[f"round_peak5_min_{d}_gbps"] for r in out
                       if r.get(f"round_peak5_min_{d}_gbps") is not None), default=None) for d in ("tx", "rx")}
    if charts:
        from . import charts as ch
        os.makedirs(os.path.join(run_dir, "report"), exist_ok=True)
        tick_list = [ticks.get(i) or tick_rates(p) for i, p in enumerate(rec_paths)]
        for p in plan["points"]:
            data = ch.chart_data(recs, tick_list, rows, p)
            if data is None:
                continue
            data["ceiling"] = ceiling.get("tx")
            data["link_max"] = link_gbps
            data["summary"] = points.get(p["id"], {})
            base = os.path.join(run_dir, "report", f"{p['id']}_traffic")
            ch.render_html(data, base + ".html")
            if "svg" in charts:
                ch.render_svg(data, base + ".svg")
    return out, points, ceiling


def report(run_dir: str, charts: str | None = None, rounds: str = "plan") -> dict:
    """Build <run>/report/. rounds="plan" uses only the rounds the plan asks for (rounds past
    it, e.g. from an earlier longer plan, stay in the raw data but are left out)."""
    manifest = read_json(os.path.join(run_dir, "manifest.json"))
    plan = manifest["plan"]
    rows = read_requests(os.path.join(run_dir, "requests.csv"))
    if rounds == "plan":
        n = plan["warmup_rounds"] + plan["measured_rounds"]
        rows = [r for r in rows if int(r["round"]) < n]
    link = manifest.get("link_max_gbps") or 200.0
    os.makedirs(os.path.join(run_dir, "report"), exist_ok=True)
    net_rounds, net_points, ceiling = analyze(run_dir, rows, plan, link, charts)

    warnings = []
    req = {p["point"]: p for p in request_summary(rows)}
    points_out = []
    for p in plan["points"]:
        pid = p["id"]
        meas = [r for r in rows if r["point"] == pid and r["warmup"] == "0"]
        want = plan["measured_rounds"] * p["concurrency"]
        if len(meas) < want:
            warnings.append(f"{pid}: {len(meas)} of {want} measured requests recorded")
        if any(r["error"] for r in meas):
            warnings.append(f"{pid}: {sum(bool(r['error']) for r in meas)} request(s) failed")
        if plan.get("force_output_length", True) and any(r.get("output_ok") == "0" for r in meas if not r["error"]):
            warnings.append(f"{pid}: some requests did not produce exactly {p['output_tokens']} output tokens")
        ok = [r for r in meas if not r["error"]]
        if not ok:
            continue
        ttft = [float(r["ttft_ms"]) for r in ok]
        points_out.append({"label": manifest["label"], "point": pid, "concurrency": p["concurrency"],
                           "input_tokens": p["input_tokens"], "output_tokens": p["output_tokens"],
                           "requests": len(ok), "ttft_ms_mean": statistics.mean(ttft),
                           "ttft_ms_p50": statistics.median(ttft), "ttft_ms_p90": _pct(ttft, 0.9),
                           "decode_tps_mean": req[pid]["decode_tps_mean"], "e2e_tps_mean": req[pid]["e2e_tps_mean"],
                           "latency_s_mean": req[pid]["latency_s_mean"],
                           "prompt_tokens_mean": req[pid]["prompt_tokens_mean"], **net_points.get(pid, {})})

    _write_csv(os.path.join(run_dir, "report", "points.csv"), points_out)
    _write_csv(os.path.join(run_dir, "report", "rounds.csv"), net_rounds)
    result = {"label": manifest["label"], "link_max_gbps": link, "ceiling_gbps": ceiling, "warnings": warnings}
    _write_md(os.path.join(run_dir, "report", "report.md"), manifest, points_out, result)
    with open(os.path.join(run_dir, "report", "report.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return {**result, "points": points_out}


def _write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


def _write_md(path: str, manifest: dict, points: list[dict], result: dict) -> None:
    f = lambda v, n=1: "—" if v in (None, "") else f"{float(v):.{n}f}"
    plan = manifest["plan"]
    lines = [f"# {manifest['campaign']} / {manifest['label']}", "",
             f"- Model: `{manifest['model']}` · endpoint `{manifest['endpoint']}` · host {manifest.get('host')}",
             f"- Devices: {', '.join(manifest.get('devices', []))} · expected: **{manifest['expect']}** · "
             f"link check: {manifest.get('link_check', 'not done')}",
             f"- Link maximum used for 'link busy': {f(result['link_max_gbps'])} Gb/s · highest 5 ms in this run: "
             f"TX ≥ {f(result['ceiling_gbps'].get('tx'))} / RX ≥ {f(result['ceiling_gbps'].get('rx'))} Gb/s",
             f"- Rounds: {plan['warmup_rounds']} warm-up + {plan['measured_rounds']} measured per point",
             f"- Notes: {manifest.get('notes') or '—'}", "",
             "| Point | Conc | In → out | TTFT mean (ms) | Decode TPS | Prefill MB/token | Prefill 1 ms p90 / p99 (Gb/s) "
             "| Transfer / gaps / stalls (s) | Link busy |",
             "|---|---|---|---|---|---|---|---|---|"]
    for p in points:
        mb = f((p.get("prefill_bytes_per_token") or 0) / 1e6, 2) if p.get("prefill_bytes_per_token") else "—"
        busy = f(100 * p["prefill_link_busy"]) + "%" if p.get("prefill_link_busy") is not None else "—"
        lines.append(f"| {p['point']} | {p['concurrency']} | {p['input_tokens']} → {p['output_tokens']} | "
                     f"{f(p['ttft_ms_mean'], 0)} | {f(p['decode_tps_mean'])} | {mb} | "
                     f"{f(p.get('prefill_1ms_p90_gbps'), 0)} / {f(p.get('prefill_1ms_p99_gbps'), 0)} | "
                     f"{f(p.get('prefill_transfer_s'), 2)} / {f(p.get('prefill_gap_s'), 2)} / "
                     f"{f(p.get('prefill_stall_s'), 2)} | {busy} |")
    lines += ["", "Means over the measured rounds (warm-up excluded). Decode TPS excludes TTFT: "
              "(output tokens − 1) ÷ (last token − first token). Prefill = request start → (last) first token. "
              "1 ms values are per sampler tick and show short collective bursts that 5 ms windows flatten. "
              "Transfer = 5 ms bins with TX ≥ 5 Gb/s; stalls = quiet runs ≥ 100 ms.", ""]
    if result["warnings"]:
        lines += ["## Warnings", ""] + [f"- {w}" for w in result["warnings"]] + [""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
