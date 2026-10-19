"""charts.py — per-point traffic charts of a run: every sample (~1 ms) and 100 ms windows, with
each round's prefill (request start -> first token) and generation (first -> last token) marked.

HTML (plotly, zoomable) is the default; SVG (matplotlib) is optional and large for long points.
"""

from __future__ import annotations

import statistics

from .rates import Recording


def chart_data(recs: list[Recording], ticks: list[dict], rows: list[dict], p: dict) -> dict | None:
    """Per-tick and 100 ms series plus phase windows for one point (seconds since it started)."""
    from bisect import bisect_left, bisect_right
    rs = [r for r in rows if r["point"] == p["id"] and not r["error"] and r.get("t_first_ns")]
    if not rs:
        return None
    t_lo = min(int(r["t0_ns"]) for r in rs)
    t_hi = max(int(r["t_end_ns"]) for r in rs)
    pick = next(((rec, tk) for rec, tk in zip(recs, ticks) if rec.span()[0] <= t_lo <= rec.span()[1]), None)
    if pick is None:
        return None
    rec, tk = pick
    t_a, t_b = t_lo - 500_000_000, t_hi + 500_000_000
    sec = lambda t: (t - t_a) / 1e9
    i, j = bisect_left(tk["t"], t_a), bisect_right(tk["t"], t_b)
    nan = float("nan")
    fix = lambda v: v if v != float("inf") else nan
    per_tick = {"x": [sec(t) for t in tk["t"][i:j]]}
    for k in ("tx", "tx_lo", "tx_hi", "rx"):
        per_tick[k] = [fix(v) for v in tk[k][i:j]]

    def series(d: str, w_ns: int, step: int):
        xs, ys = [], []
        t = t_a
        while t <= t_b:
            parts = []
            for dev in rec.devices:
                ser = rec.state.series[(dev, d)]
                i = ser.index_at(t)
                r = ser.window(i, w_ns) if i >= 0 and t - ser.t[i] < 5_000_000 else None   # no data past the recording
                if r is None:
                    parts = None
                    break
                parts.append(r.gbps)
            xs.append(sec(t))
            ys.append(sum(parts) if parts else nan)
            t += step
        return xs, ys

    rounds = []
    for rnd in sorted({int(r["round"]) for r in rs}):
        rr = [r for r in rs if int(r["round"]) == rnd]
        ttft = [float(r["ttft_ms"]) / 1000 for r in rr if r.get("ttft_ms")]
        tps = [float(r["decode_tps"]) for r in rr if r.get("decode_tps")]
        rounds.append({"round": rnd, "warmup": rr[0]["warmup"] == "1",
                       "start": sec(min(int(r["t0_ns"]) for r in rr)),
                       "first": sec(max(int(r["t_first_ns"]) for r in rr)),
                       "end": sec(max(int(r["t_end_ns"]) for r in rr)),
                       "ttft": (min(ttft), max(ttft)) if ttft else None,
                       "tps": statistics.mean(tps) if tps else None,
                       "firsts": sorted(sec(int(r["t_first_ns"])) for r in rr)})
    return {"point": p, "tick": per_tick, "tx100": series("tx", 100_000_000, 50_000_000),
            "rx100": series("rx", 100_000_000, 50_000_000), "rounds": rounds}


def _labels(rd: dict) -> tuple[str, str]:
    lo, hi = rd["ttft"] or (None, None)
    ttft = "—" if lo is None else (f"TTFT {lo:.2f} s" if hi - lo < 0.005 else f"TTFT {lo:.2f}–{hi:.2f} s")
    tps = "—" if rd["tps"] is None else f"decode {rd['tps']:.1f} tok/s" + (" per request" if rd["ttft"] and hi - lo >= 0.005 else "")
    return ttft, tps


def _zoom_max(data: dict) -> float:
    """y-range for the decode panel: 3 × the 95th percentile of 100 ms TX during generation."""
    xs, ys = data["tx100"]
    vals = sorted(y for x, y in zip(xs, ys) if y == y and
                  any(rd["first"] + 0.2 <= x <= rd["end"] for rd in data["rounds"]))
    p95 = vals[int(0.95 * (len(vals) - 1))] if vals else 0.0
    return max(1.0, 3 * p95)


def _summary_text(data: dict) -> str:
    s = data["summary"]
    f = lambda k, fmt: "—" if s.get(k) is None else format(s[k], fmt)
    return (f"Measured rounds (mean): prefill {f('prefill_s_mean', '.2f')} s · "
            f"{f('prefill_mean_gbps', '.1f')} Gb/s mean during prefill · 5 ms peak ≥ {f('prefill_peak5_min_gbps', '.1f')} Gb/s · "
            f"link busy {('—' if s.get('prefill_link_busy') is None else format(100 * s['prefill_link_busy'], '.0f') + '%')} · "
            f"{('—' if s.get('prefill_bytes_per_token') is None else format(s['prefill_bytes_per_token'] / 1e6, '.2f'))} MB per input token · "
            f"generation {f('generation_mean_gbps', '.2f')} Gb/s mean")


def _title(data: dict) -> str:
    p = data["point"]
    return f"{p['id']}: concurrency {p['concurrency']}, {p['input_tokens']} input → {p['output_tokens']} output tokens"


def render_svg(data: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.transforms import blended_transform_factory

    plt.rcParams.update({"path.simplify": False,      # keep every sample: zooming into the SVG shows them
                         "svg.fonttype": "none", "agg.path.chunksize": 0})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2], "hspace": 0.16})
    ref = max(v for v in (data["link_max"], data["ceiling"], 1.0) if v)
    top = ref * 1.3                                     # room for 1 ms values above the ceiling (timing noise)
    zoom = _zoom_max(data)
    for ax in (ax1, ax2):
        for rd in data["rounds"]:
            ax.axvspan(rd["start"], rd["first"], color="#9aa0a6" if rd["warmup"] else "#2da44e",
                       alpha=0.28 if rd["warmup"] else 0.22, lw=0)
            ax.axvspan(rd["first"], rd["end"], color="#0969da", alpha=0.08, lw=0)
        ax.grid(alpha=0.25)
    tk = data["tick"]
    ax1.fill_between(tk["x"], tk["tx_lo"], tk["tx_hi"], color="#e8590c", alpha=0.18, lw=0, step=None)
    ax1.plot(tk["x"], tk["tx"], color="#e8590c", lw=0.35)
    for ax in (ax1, ax2):
        ax.plot(*data["tx100"], color="#7c2d12", lw=1.8)
        ax.plot(*data["rx100"], color="#1c7ed6", lw=1.1, ls="--", alpha=0.9)
    if data["link_max"]:
        ax1.axhline(data["link_max"], color="#cf222e", ls="--", lw=1.1)
        ax1.text(1.004, data["link_max"], f"link maximum\n{data['link_max']:.0f} Gb/s",
                 transform=ax1.get_yaxis_transform(), va="bottom", ha="left", color="#cf222e", fontsize=8.5)
    if data["ceiling"]:
        ax1.axhline(data["ceiling"], color="#8250df", ls=":", lw=1.2)
        ax1.text(1.004, data["ceiling"], f"highest 5 ms\nin this run\n≥ {data['ceiling']:.1f} Gb/s",
                 transform=ax1.get_yaxis_transform(), va="top", ha="left", color="#8250df", fontsize=8.5)
    ax1.set_ylim(0, top)
    ax2.set_ylim(0, zoom)
    above = blended_transform_factory(ax1.transData, ax1.transAxes)
    for rd in data["rounds"]:
        if len(rd["firsts"]) > 1:
            for x in rd["firsts"]:
                ax1.plot([x, x], [0.88, 0.98], transform=above, color="#1a7f37", lw=1.4,
                         alpha=0.5 if rd["warmup"] else 0.9)
        ttft, tps = _labels(rd)
        grey = rd["warmup"]
        ax1.text((rd["start"] + rd["first"]) / 2, 1.01, ("warm-up\n" if grey else "") + ttft, transform=above,
                 ha="center", va="bottom", fontsize=8.5, color="#57606a" if grey else "#1a7f37", fontweight="bold")
        ax2.text((rd["first"] + rd["end"]) / 2, zoom * 0.95, tps, ha="center", va="top", fontsize=8.5,
                 color="#57606a" if grey else "#0550ae", fontweight="bold")
    ax1.set_ylabel("Gb/s (full scale)")
    ax2.set_ylabel(f"Gb/s (zoom 0–{zoom:.1f})")
    ax2.set_xlabel("seconds since the point started")
    ax1.set_title(_title(data), fontsize=12, loc="left", pad=34)
    ax2.set_title("Zoom on generation traffic (100 ms windows)", fontsize=9, loc="left")
    handles = [plt.Line2D([], [], color="#e8590c", lw=0.8, label="TX, every sample (~1 ms)"),
               Patch(color="#e8590c", alpha=0.25, label="its timing bound (read time ± ~0.1–0.2 ms)"),
               plt.Line2D([], [], color="#7c2d12", lw=1.8, label="TX, 100 ms windows"),
               plt.Line2D([], [], color="#1c7ed6", lw=1.1, ls="--", label="RX, 100 ms windows"),
               Patch(color="#2da44e", alpha=0.3, label="prefill: request start → first token (TTFT)"),
               Patch(color="#0969da", alpha=0.15, label="generation: first → last token"),
               Patch(color="#9aa0a6", alpha=0.35, label="warm-up (not measured)")]
    if any(len(rd["firsts"]) > 1 for rd in data["rounds"]):
        handles.append(plt.Line2D([], [], color="#1a7f37", lw=1.4, label="first token of each request"))
    fig.subplots_adjust(bottom=0.16)
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.035))
    fig.text(0.01, 0.005, _summary_text(data), fontsize=9, color="#24292f")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    # Lines keep their on-screen width at any zoom (otherwise a 100× zoom draws 100× thicker lines).
    with open(path, encoding="utf-8") as fh:
        svg = fh.read()
    svg = svg.replace('<style type="text/css">', '<style type="text/css">path{vector-effect: non-scaling-stroke} ', 1)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(svg)


def render_html(data: dict, path: str) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.6, 0.4],
                        subplot_titles=("Full scale", "Zoom: generation traffic"))
    ref = max(v for v in (data["link_max"], data["ceiling"], 1.0) if v)
    zoom = _zoom_max(data)
    tk = data["tick"]
    fig.add_trace(go.Scattergl(x=tk["x"], y=tk["tx_hi"], mode="lines", line=dict(width=0), hoverinfo="skip",
                               showlegend=False), row=1, col=1)
    fig.add_trace(go.Scattergl(x=tk["x"], y=tk["tx_lo"], mode="lines", line=dict(width=0), fill="tonexty",
                               fillcolor="rgba(232,89,12,0.18)", name="timing bound", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scattergl(x=tk["x"], y=tk["tx"], mode="lines", name="TX, every sample (~1 ms)",
                               line=dict(color="#e8590c", width=0.8)), row=1, col=1)
    for row in (1, 2):
        show = row == 1
        fig.add_trace(go.Scattergl(x=data["tx100"][0], y=data["tx100"][1], name="TX, 100 ms", legendgroup="tx100",
                                   showlegend=show, mode="lines", line=dict(color="#7c2d12", width=2.5)), row=row, col=1)
        fig.add_trace(go.Scattergl(x=data["rx100"][0], y=data["rx100"][1], name="RX, 100 ms", legendgroup="rx100",
                                   showlegend=show, mode="lines", line=dict(color="#1c7ed6", width=1.2, dash="dash")),
                      row=row, col=1)
        for rd in data["rounds"]:
            ttft, tps = _labels(rd)
            fig.add_vrect(x0=rd["start"], x1=rd["first"], fillcolor="#9aa0a6" if rd["warmup"] else "#2da44e",
                          opacity=0.22, line_width=0, row=row, col=1,
                          annotation_text=(("warm-up · " if rd["warmup"] else "") + ttft) if row == 1 else None,
                          annotation_position="top", annotation_font_size=10)
            fig.add_vrect(x0=rd["first"], x1=rd["end"], fillcolor="#0969da", opacity=0.07, line_width=0,
                          row=row, col=1, annotation_text=tps if row == 2 else None,
                          annotation_position="top", annotation_font_size=10)
    for rd in data["rounds"]:
        if len(rd["firsts"]) > 1:
            for x in rd["firsts"]:
                fig.add_vline(x=x, line=dict(color="#1a7f37", width=1, dash="dot"), row=1, col=1)
    if data["link_max"]:
        fig.add_hline(y=data["link_max"], line=dict(color="#cf222e", dash="dash", width=1), row=1, col=1,
                      annotation_text=f"link maximum {data['link_max']:.0f} Gb/s", annotation_position="top left")
    if data["ceiling"]:
        fig.add_hline(y=data["ceiling"], line=dict(color="#8250df", dash="dot", width=1), row=1, col=1,
                      annotation_text=f"highest 5 ms in this run ≥ {data['ceiling']:.1f} Gb/s",
                      annotation_position="bottom left")
    fig.update_yaxes(range=[0, ref * 1.3], title_text="Gb/s", row=1, col=1)
    fig.update_yaxes(range=[0, zoom], title_text="Gb/s", row=2, col=1)
    fig.update_xaxes(title_text="seconds since the point started", row=2, col=1)
    fig.update_layout(title=_title(data) + "<br><sup>" + _summary_text(data) + "</sup>", template="plotly_white",
                      height=850, width=1500, legend=dict(orientation="h", y=-0.1, x=0))
    fig.write_html(path, include_plotlyjs=True)
