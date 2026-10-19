"""compare.py — compare two runs point by point (e.g. a 200G and a 100G link, two settings).

Uses each run's report/points.csv (built by `netinfermon report`). Per point: TTFT and decode TPS
change, and the prefill split, so a TTFT change can be told apart: transfer time is the part
the link explains; stall time is whole-cluster waiting that can differ from run to run.
"""

from __future__ import annotations

import csv
import json
import os

from .loadgen import read_json


def load_points(run_dir: str) -> tuple[dict, dict]:
    manifest = read_json(os.path.join(run_dir, "manifest.json"))
    path = os.path.join(run_dir, "report", "points.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run `netinfermon report` for this run first")
    with open(path, newline="", encoding="utf-8") as f:
        return manifest, {r["point"]: r for r in csv.DictReader(f)}


def num(v):
    return None if v in (None, "") else float(v)


def _sum(row: dict, *keys):
    vals = [num(row.get(k)) for k in keys]
    return None if any(v is None for v in vals) else sum(vals)


def change(a, b):
    return None if a in (None, 0) or b is None else 100.0 * (b - a) / a


def compare(base_dir: str, test_dir: str, out_dir: str | None = None) -> dict:
    mb, pb = load_points(base_dir)
    mt, pt = load_points(test_dir)
    a, b = mb["label"], mt["label"]
    warnings = []
    if mb["model"] != mt["model"]:
        warnings.append(f"different models: {mb['model']} vs {mt['model']}")
    if mb["plan"].get("seed") != mt["plan"].get("seed"):
        warnings.append("different plan seeds: the two runs did not get the same prompts")
    rows = []
    for pid, x in pb.items():
        y = pt.get(pid)
        if not y:
            continue
        if any(x[k] != y[k] for k in ("concurrency", "input_tokens", "output_tokens")):
            warnings.append(f"{pid}: different workloads ({x['concurrency']}/{x['input_tokens']}/{x['output_tokens']} "
                            f"vs {y['concurrency']}/{y['input_tokens']}/{y['output_tokens']}); not comparable")
            continue
        rows.append({
            "point": pid, "concurrency": int(x["concurrency"]), "input_tokens": int(x["input_tokens"]),
            "output_tokens": int(x["output_tokens"]),
            f"ttft_ms_{a}": num(x["ttft_ms_mean"]), f"ttft_ms_{b}": num(y["ttft_ms_mean"]),
            "ttft_change_%": change(num(x["ttft_ms_mean"]), num(y["ttft_ms_mean"])),
            f"decode_tps_{a}": num(x["decode_tps_mean"]), f"decode_tps_{b}": num(y["decode_tps_mean"]),
            "decode_tps_change_%": change(num(x["decode_tps_mean"]), num(y["decode_tps_mean"])),
            f"transfer_s_{a}": num(x.get("prefill_transfer_s")), f"transfer_s_{b}": num(y.get("prefill_transfer_s")),
            "transfer_change_s": (None if None in (num(x.get("prefill_transfer_s")), num(y.get("prefill_transfer_s")))
                                  else num(y["prefill_transfer_s"]) - num(x["prefill_transfer_s"])),
            f"stall_s_{a}": num(x.get("prefill_stall_s")), f"stall_s_{b}": num(y.get("prefill_stall_s")),
            f"leadin_tail_s_{a}": _sum(x, "prefill_leadin_s", "prefill_tail_s"),
            f"leadin_tail_s_{b}": _sum(y, "prefill_leadin_s", "prefill_tail_s"),
            f"prefill_1ms_p99_{a}": num(x.get("prefill_1ms_p99_gbps")),
            f"prefill_1ms_p99_{b}": num(y.get("prefill_1ms_p99_gbps")),
        })
    result = {"base": a, "test": b, "model": mb["model"], "rows": rows, "warnings": warnings}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if rows:
            with open(os.path.join(out_dir, "comparison.csv"), "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        with open(os.path.join(out_dir, "comparison.md"), "w", encoding="utf-8") as fh:
            fh.write(markdown(result))
    return result


def markdown(res: dict) -> str:
    a, b = res["base"], res["test"]
    g = lambda v: "—" if v is None else f"{v:+.1f}%"
    h = lambda v, n=0: "—" if v is None else f"{v:.{n}f}"
    md = [f"# {a} → {b}", "", f"- Model: `{res['model']}`", "",
          f"| Point | Conc | In → out | TTFT {a} (ms) | TTFT {b} (ms) | TTFT change | Decode TPS change "
          f"| Prefill transfer change (s) | Stalls {a} / {b} (s) | Lead-in + tail {a} / {b} (s) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        md.append(f"| {r['point']} | {r['concurrency']} | {r['input_tokens']} → {r['output_tokens']} | "
                  f"{h(r[f'ttft_ms_{a}'])} | {h(r[f'ttft_ms_{b}'])} | {g(r['ttft_change_%'])} | "
                  f"{g(r['decode_tps_change_%'])} | "
                  f"{'—' if r['transfer_change_s'] is None else format(r['transfer_change_s'], '+.2f')} | "
                  f"{h(r[f'stall_s_{a}'], 2)} / {h(r[f'stall_s_{b}'], 2)} | "
                  f"{h(r[f'leadin_tail_s_{a}'], 2)} / {h(r[f'leadin_tail_s_{b}'], 2)} |")
    md += ["", "Means over measured rounds. Transfer = prefill time with the link carrying data (5 ms bins "
           "≥ 5 Gb/s): its change is the part of the TTFT change the link explains. Stalls (quiet runs "
           "≥ 100 ms between transfers) are cluster-wide waiting. Lead-in + tail = quiet time before the "
           "first and after the last transfer (request handling, final compute). Both can differ between "
           "runs for reasons other than the link.", ""]
    if res["warnings"]:
        md += ["**Warnings:**", ""] + [f"- {w}" for w in res["warnings"]] + [""]
    return "\n".join(md)
