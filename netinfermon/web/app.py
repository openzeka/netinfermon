"""app.py — netinfermon web page: Live traffic, Benchmark, Results, Compare.

The sampler, recordings, runs and the control API live in monitor.py (one instance per process,
started by `netinfermon serve` before this page). Rate maths is in rates.py.
"""

from __future__ import annotations

import glob
import io
import json
import os
import time
import zipfile
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from netinfermon import analysis, compare, loadgen
from netinfermon.monitor import get_service, restart_service
from netinfermon.rates import DIRS, HIST_LABELS, WINDOWS_MS, Recording, summarize

REFRESH_S = 0.5
CHART_LIVE = 2400               # live view: last 2 minutes (one point per 50 ms)
COLOR = {"tx": "#ff9f43", "rx": "#00d4ff"}
EXPECT = {"all": "every device (full link)", "one": "one device only", "any": "don't check"}

st.set_page_config(page_title="netinfermon", page_icon="📶", layout="wide")
svc = get_service()
S = svc.settings


# --- sidebar: recording and settings -----------------------------------------------------

with st.sidebar:
    st.markdown("## Recording")
    if svc.rec_file and svc.rec_owner == "run":
        st.info(f"Recording for the running benchmark · {time.time() - svc.rec_started:.0f} s")
    elif svc.rec_file and svc.rec_owner == "api":
        st.info(f"Recording started through the API · {time.time() - svc.rec_started:.0f} s\n\n`{svc.rec_path}`")
    elif svc.rec_file:
        st.success(f"Recording · {time.time() - svc.rec_started:.0f} s")
        label = st.text_input("Marker name", placeholder="e.g. request 1")
        if st.button("📍 Add marker", width="stretch"):
            svc.marker(label)
            st.toast(f"Marker: {label or 'marker'}")
        if st.button("⏹ Stop recording", width="stretch"):
            svc.stop_recording()
            st.rerun()
    else:
        if st.button("⏺ Start recording", width="stretch"):
            try:
                svc.start_recording()
            except RuntimeError as e:
                st.error(str(e))
            st.rerun()
    st.caption("Benchmark runs record by themselves. Manual recordings appear under Results → Manual recordings.")

    if not svc.alive:
        st.error("The sampler stopped.\n\n" + "\n".join(svc.stderr))
        if st.button("↻ Restart", width="stretch"):
            restart_service()
            st.rerun()

    with st.expander("⚙️ Settings"):
        link_gbps = st.number_input("Link maximum (Gb/s)", min_value=1.0, max_value=3200.0,
                                    value=float(svc.link_gbps), step=1.0,
                                    help="Highest rate the link reaches, per direction, all devices together. "
                                         "Set `[link] max_gbps` in ~/.config/netinfermon/config.toml to the measured value.")
        target_gbps = st.number_input("Compare with link speed (Gb/s)", min_value=1.0, max_value=3200.0,
                                      value=round(link_gbps / 2 / 5) * 5.0, step=5.0,
                                      help="For the 'If the link were …' estimate. Lower = more pessimistic.")
        if st.button("Reset", width="stretch", disabled=bool(svc.rec_file),
                     help="Clears averages and peaks. Starting a recording also does this."):
            with svc.lock:
                svc.state.reset_peaks()
            st.rerun()
    with svc.lock:
        svc.state.link_gbps = link_gbps


def peak_txt(p) -> str:
    return "—" if p is None else f"≥ {p[0].lo:.1f} Gb/s"


def avg_txt(a) -> str:
    return "—" if a is None else f"{a['gbps']:.2f} Gb/s"


def runs() -> list[str]:
    """Run directories (campaign/label), newest first."""
    paths = glob.glob(os.path.join(S.data_dir, "*", "*", "manifest.json"))
    paths.sort(key=os.path.getmtime, reverse=True)
    return [os.path.relpath(os.path.dirname(p), S.data_dir) for p in paths]


st.markdown("## 📶 netinfermon — inference & RDMA traffic monitor")
tab_live, tab_bench, tab_results, tab_compare = st.tabs(["Live", "Benchmark", "Results", "Compare"])


# --- Live -------------------------------------------------------------------------------

with tab_live:
    @st.fragment(run_every=REFRESH_S)
    def live_panel():
        with svc.lock:
            state = svc.state
            recording = bool(svc.rec_file)
            rec_s = time.time() - svc.rec_started if recording else None
            frozen = state.trackers["overall"].frozen
            o_label, p_label = state.trackers["overall"].label, state.trackers["phase"].label
            avg = {(n, d): state.average(n, d) for n in ("overall", "phase") for d in DIRS}
            pk = {(n, d): state.peak(n, d, WINDOWS_MS[0]) for n in ("overall", "phase") for d in DIRS}
            link = {n: state.link_stats(n, target_gbps) for n in ("overall", "phase")}
            health = state.health()
            devices = list(state.devices)
        num = lambda v, f: "—" if v is None else format(v, f)
        rec_txt = (f"⏺ recording {int(rec_s // 60):02d}:{int(rec_s % 60):02d}" if recording
                   else ("⏹ recording stopped (values frozen)" if frozen else "not recording"))
        st.caption(f"Devices: {', '.join(devices) or '—'} (summed) · TX = sent, RX = received · sampling every "
                   f"**{num(health['tick_p50_ms'], '.2f')} ms** · **{rec_txt}** · phase: **{p_label}**")
        for n, title in (("overall", f"Overall — {o_label}"), ("phase", f"This phase — {p_label}")):
            st.markdown(f"#### {title}")
            c = st.columns(4)
            for i, d in enumerate(DIRS):
                a = avg[(n, d)]
                c[i].metric(f"Average {d.upper()}", avg_txt(a),
                            None if a is None else f"{a['GB']:.1f} GB · {a['s']:.0f} s", delta_color="off")
                p = pk[(n, d)]
                c[2 + i].metric(f"Peak {d.upper()} ({WINDOWS_MS[0]} ms)", peak_txt(p),
                                None if p is None else f"in: {p[1]}", delta_color="off")
        st.markdown(f"#### If the link were {target_gbps:g} Gb/s instead of {link_gbps:g} Gb/s")
        c = st.columns(4)
        for i, (n, title) in enumerate((("phase", "This phase"), ("overall", "Overall"))):
            e = link[n]
            c[2 * i].metric(f"{title}: time would grow by", "—" if e is None else f"+{100 * (e['factor'] - 1):.1f}%",
                            None if e is None else f"{e['active_s']:.2f} s → {e['active_s'] * e['factor']:.2f} s (at most)",
                            delta_color="off")
            c[2 * i + 1].metric(f"{title}: link busy (f)", "—" if e is None else f"{100 * e['f']:.1f}%",
                                delta_color="off")
        st.caption("A rough upper bound from 5 ms windows; it understates collectives shorter than 5 ms. "
                   "For a real answer, run the same benchmark on both links and use Compare.")

    live_panel()

    paused = st.session_state.get("chart_hold") is not None

    @st.fragment(run_every=None if paused else REFRESH_S)
    def chart_panel():
        c1, c2 = st.columns([1, 5])
        if paused:
            if c1.button("▶ Release (back to live)", width="stretch"):
                st.session_state.chart_hold = None
                st.rerun()
            held_at, chart, reqs = st.session_state.chart_hold
            c2.caption(f"Held at {held_at:%H:%M:%S}. Drag to zoom, double-click to reset; up to 10 minutes of history.")
        else:
            if c1.button("⏸ Hold", width="stretch"):
                with svc.lock:
                    st.session_state.chart_hold = (datetime.now(), list(svc.chart), request_spans())
                st.rerun()
            with svc.lock:
                chart = list(svc.chart)[-CHART_LIVE:]
            reqs = request_spans()
            c2.caption("Live, last 2 minutes. Green = prefill of a benchmark request (start → first token), "
                       "blue = its generation. Hold to stop scrolling and look back.")
        fig = go.Figure()
        if chart:
            xs = [p["t"] for p in chart]
            for d in DIRS:
                fig.add_trace(go.Scattergl(x=xs, y=[p[f"{d}{WINDOWS_MS[0]}"] for p in chart],
                                           name=f"{d.upper()} {WINDOWS_MS[0]} ms", mode="lines",
                                           line=dict(color=COLOR[d], width=1), opacity=0.45))
                fig.add_trace(go.Scattergl(x=xs, y=[p[f"{d}100"] for p in chart], name=f"{d.upper()} 100 ms",
                                           mode="lines", line=dict(color=COLOR[d], width=2.5)))
            t_lo = chart[0]["t"]
            for r in reqs:
                if r["end"] < t_lo:
                    continue
                fig.add_vrect(x0=r["start"], x1=r["first"], fillcolor="#2da44e", opacity=0.18, line_width=0)
                if r["gen_end"]:
                    fig.add_vrect(x0=r["first"], x1=r["gen_end"], fillcolor="#0969da", opacity=0.06, line_width=0)
        fig.update_layout(template="plotly_dark", height=340, margin=dict(l=10, r=10, t=30, b=10),
                          hovermode="x unified", yaxis_title="Gb/s",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                          dragmode="zoom" if paused else False, uirevision="held" if paused else None)
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig, width="stretch", theme=None,
                        config={"displayModeBar": paused, "scrollZoom": paused, "displaylogo": False})

    def request_spans() -> list[dict]:
        """Benchmark requests of the last few minutes as wall-clock spans (prefill, generation)."""
        if not svc.runner:
            return []
        done, inflight = svc.runner.timeline()
        with svc.lock:
            wall = svc.state.wall
            now_t = svc.state.last_t()
        dt = lambda t: datetime.fromtimestamp(wall(int(t)))
        out = []
        for r in list(done)[-60:]:
            if r.get("t_first_ns"):
                out.append({"start": dt(r["t0_ns"]), "first": dt(r["t_first_ns"]), "gen_end": dt(r["t_end_ns"]),
                            "end": dt(r["t_end_ns"])})
        for r in inflight:
            first = r["t_first"] or now_t
            out.append({"start": dt(r["t0"]), "first": dt(first), "gen_end": dt(now_t) if r["t_first"] else None,
                        "end": dt(now_t)})
        return out

    chart_panel()

    with st.expander("Details: per device, measurement health"):
        @st.fragment(run_every=2.0)
        def details_panel():
            with svc.lock:
                state = svc.state
                dev_rows = [{"device": dev, "direction": d.upper(),
                             "average Gb/s": "—" if (a := state.average("overall", d, dev)) is None else f"{a['gbps']:.2f}",
                             **{f"peak {w} ms": peak_txt(state.peak("overall", d, w, dev)) for w in WINDOWS_MS}}
                            for dev in state.devices for d in DIRS]
                health = state.health()
            st.dataframe(pd.DataFrame(dev_rows), width="stretch", hide_index=True)
            num = lambda v, f: "—" if v is None else format(v, f)
            st.markdown(
                f"**Measurement health:** sampler {'🟢' if svc.alive else '🔴'} · tick p50 "
                f"{num(health['tick_p50_ms'], '.2f')} / p95 {num(health['tick_p95_ms'], '.2f')} ms · read "
                f"half-width p95 {num(health['halfw_p95_us'], '.0f')} µs · read errors {health['errors']} · "
                f"counter resets {health['resets']} · impossible values {health['impossible']} (should be 0)")
            st.caption("Counters: /sys/class/infiniband/<device>/ports/1/counters/port_xmit_data and "
                       "port_rcv_data (RDMA bytes ÷ 4). Ordinary tools (nload, ip -s link) do not see RDMA traffic.")

        details_panel()


# --- Benchmark --------------------------------------------------------------------------

with tab_bench:
    left, right = st.columns([2, 3])
    with left:
        st.markdown("#### New run")
        endpoint = st.text_input("Endpoint", value=st.session_state.get("endpoint", S.endpoint))
        st.session_state.endpoint = endpoint
        if st.button("Get models") or "models" not in st.session_state:
            try:
                st.session_state.models = [m["id"] for m in loadgen.get_models(
                    endpoint, timeout=5, verify_tls=not S.insecure_tls).get("data", [])]
            except Exception as e:                               # noqa: BLE001
                st.session_state.models = []
                st.caption(f"Could not list models at {endpoint}: {e}")
        model = st.selectbox("Model", st.session_state.models or ["(none found)"])
        mode = st.radio("Workload", ["Built-in plan", "Custom"], horizontal=True)
        if mode == "Built-in plan":
            plans = loadgen.list_plans()
            names = list(plans)
            name = st.selectbox("Plan", names, index=names.index("smoke") if "smoke" in names else 0,
                                format_func=lambda n: f"{n} — {plans[n][:70]}")
            plan = loadgen.load_plan(name)
            st.dataframe(pd.DataFrame(plan["points"]), hide_index=True, width="stretch")
            st.caption(f"{plan['warmup_rounds']} warm-up + {plan['measured_rounds']} measured rounds per point, "
                       f"seed {plan['seed']}")
        else:
            c = st.columns(2)
            ins = c[0].text_input("Input tokens (comma-separated)", "1024, 8192")
            out_tok = c[1].number_input("Output tokens", 1, 32768, 1024, step=128)
            concs = c[0].text_input("Concurrency (comma-separated)", "1, 4")
            seed = c[1].number_input("Seed", 0, 2**31 - 1, 0)
            warm = c[0].number_input("Warm-up rounds", 0, 10, 1)
            meas = c[1].number_input("Measured rounds", 1, 100, 3)
            force = st.checkbox("Force exact output length", True,
                                help="ignore_eos + min_tokens (vLLM, SGLang) so every request decodes the same amount")
            try:
                plan = loadgen.build_plan([int(x) for x in ins.split(",")], int(out_tok),
                                          [int(x) for x in concs.split(",")], int(warm), int(meas), int(seed), force)
            except ValueError:
                plan = None
                st.error("Input tokens and concurrency must be comma-separated whole numbers.")
        c = st.columns(2)
        campaign = c[0].text_input("Campaign", st.session_state.get("campaign", "default"),
                                   help="Runs to compare belong to one campaign (same prompts).")
        label = c[1].text_input("Run label", placeholder="e.g. 200G")
        st.session_state.campaign = campaign
        expect = st.selectbox("Link check", list(EXPECT), format_func=EXPECT.get,
                              help="Checked after the first round: is the traffic on every device, or on one only?")
        notes = st.text_input("Notes", placeholder="what is special about this run (link, settings, …)")
        running = bool(svc.run_thread and svc.run_thread.is_alive())
        if st.button("▶ Start run", type="primary", disabled=running or not label or plan is None
                     or not st.session_state.models, width="stretch"):
            try:
                svc.start_run(campaign, label, plan, endpoint, model, expect, notes or None)
                st.rerun()
            except Exception as e:                               # noqa: BLE001
                st.error(str(e))
        st.caption("A label that already exists continues where it stopped (recorded rounds are skipped).")

    with right:
        @st.fragment(run_every=1.0)
        def run_panel():
            stt = svc.run_status()
            if not stt:
                st.info("No run yet.")
                return
            prog = stt["progress"]
            st.markdown(f"#### {stt['campaign']} / {stt['label']} — **{stt['state']}**")
            if prog.get("total"):
                st.progress(prog["done"] / prog["total"], text=f"{prog['done']}/{prog['total']} rounds · {prog['current']}")
            if stt["error"]:
                st.error(stt["error"])
            if svc.run_thread and svc.run_thread.is_alive():
                if st.button("⏹ Stop after this round"):
                    svc.stop_run()
            rows = loadgen.read_requests(os.path.join(stt["dir"], "requests.csv"))
            summ = loadgen.request_summary(rows)
            if summ:
                st.dataframe(pd.DataFrame(summ), hide_index=True, width="stretch",
                             column_config={k: st.column_config.NumberColumn(k, format="%.1f")
                                            for k in ("ttft_ms_mean", "decode_tps_mean", "e2e_tps_mean",
                                                      "latency_s_mean", "prompt_tokens_mean")})
            st.code("\n".join(stt["log"]) or "—", language=None)

        run_panel()


# --- Results ----------------------------------------------------------------------------

with tab_results:
    # Refreshes by itself, so runs started or finished meanwhile (also from the CLI) show up
    @st.fragment(run_every=5.0)
    def results_panel():
        all_runs = runs()
        if not all_runs:
            st.caption(f"No runs yet in {S.data_dir}.")
        else:
            choice = st.selectbox("Run (campaign / label)", all_runs)
            rdir = os.path.join(S.data_dir, choice)
            man = loadgen.read_json(os.path.join(rdir, "manifest.json"))
            st.caption(f"Model `{man['model']}` · {man['endpoint']} · expected link: {man['expect']} · "
                       f"link check: {man.get('link_check', '—')} · notes: {man.get('notes') or '—'}")
            busy = bool(svc.run_thread and svc.run_thread.is_alive() and svc.runner and svc.runner.dir == rdir)
            c = st.columns(3)
            charts = c[0].selectbox("Charts", ["none", "html", "html+svg"],
                                    help="Per-point traffic charts. SVG files are large for long points.")
            if c[1].button("Build report", disabled=busy, help="Reads the whole recording; takes a while for long runs."):
                with st.spinner("Building the report…"):
                    analysis.report(rdir, None if charts == "none" else charts)
            rep = os.path.join(rdir, "report", "report.md")
            if os.path.exists(rep):
                with open(rep, encoding="utf-8") as fh:
                    st.markdown(fh.read())
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for fn in ("manifest.json", "models.json", "requests.csv"):
                        if os.path.exists(os.path.join(rdir, fn)):
                            z.write(os.path.join(rdir, fn), fn)
                    for p in glob.glob(os.path.join(rdir, "report", "*")):
                        z.write(p, os.path.join("report", os.path.basename(p)))
                c[2].download_button("Download report (zip)", buf.getvalue(),
                                     file_name=f"{choice.replace(os.sep, '_')}_report.zip")
                st.caption("Raw recordings are not in the zip (they are large); copy them with `netinfermon fetch`.")
            else:
                summ = loadgen.request_summary(loadgen.read_requests(os.path.join(rdir, "requests.csv")))
                if summ:
                    st.markdown("Request numbers so far (build the report for the traffic side):")
                    st.dataframe(pd.DataFrame(summ), hide_index=True, width="stretch")

        st.divider()
        st.markdown("#### Manual recordings")
        files = sorted(glob.glob(os.path.join(S.recordings_dir, "rec-*.csv")), key=os.path.getmtime, reverse=True)
        if not files:
            st.caption("None yet.")
        else:
            rec_choice = st.selectbox("Recording", files, format_func=os.path.basename)
            if st.button("Summarize per marker"):
                if rec_choice == svc.rec_path and svc.rec_file:
                    with svc.lock:
                        svc.rec_file.flush()
                with st.spinner("Calculating…"):
                    st.session_state.summary = (rec_choice, pd.DataFrame(summarize(
                        Recording(rec_choice, link_gbps), link_gbps=link_gbps, target_gbps=target_gbps)))
            if st.session_state.get("summary") and st.session_state.summary[0] == rec_choice:
                df = st.session_state.summary[1]
                key = ["phase", "direction", "duration_s", "GB", "mean_Gbps", *[f"{w}ms_peak_min" for w in WINDOWS_MS],
                       "active_span_s", "link_busy_%", f"%time_{HIST_LABELS[-1]}"]
                show = df if st.checkbox("All columns") else df[[k for k in key if k in df.columns]]
                st.dataframe(show, width="stretch", hide_index=True)
                st.download_button("Summary CSV", df.to_csv(index=False).encode(),
                                   file_name=os.path.basename(rec_choice).replace(".csv", "-summary.csv"))

    results_panel()


# --- Compare ----------------------------------------------------------------------------

with tab_compare:
    # Refreshes by itself, so newly built reports show up
    @st.fragment(run_every=5.0)
    def compare_panel():
        reported = [r for r in runs() if os.path.exists(os.path.join(S.data_dir, r, "report", "points.csv"))]
        if len(reported) < 2:
            st.caption("Needs two runs with a built report (Results tab → Build report).")
        else:
            c = st.columns(2)
            base = c[0].selectbox("Base run", reported, index=min(1, len(reported) - 1))
            test = c[1].selectbox("Compared run", reported, index=0)
            if base != test:
                res = compare.compare(os.path.join(S.data_dir, base), os.path.join(S.data_dir, test))
                st.markdown(compare.markdown(res))
                if res["rows"]:
                    st.download_button("Comparison CSV", pd.DataFrame(res["rows"]).to_csv(index=False).encode(),
                                       file_name="comparison.csv")

    compare_panel()
