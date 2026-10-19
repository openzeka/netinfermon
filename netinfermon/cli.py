"""netinfermon command line.

    netinfermon serve                       service + web page (http://<host>:8502)
    netinfermon doctor                      check RDMA devices, counters, sampler, ports
    netinfermon plans                       list built-in benchmark plans
    netinfermon run CAMPAIGN LABEL --plan linkspeed [--endpoint URL] [--model ID] [--expect all|one|any] [--wait]
    netinfermon status | stop               current run
    netinfermon record CAMPAIGN LABEL -- COMMAND...   record traffic while any command runs
    netinfermon summarize RECORDING.csv     per-marker summary of a recording
    netinfermon report CAMPAIGN/LABEL [--charts html|html+svg]
    netinfermon compare CAMPAIGN/BASE CAMPAIGN/TEST
    netinfermon fetch [USER@]HOST [CAMPAIGN]  copy runs from a node (rsync over ssh)
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from . import __version__, config, loadgen, sampler

HERE = os.path.dirname(os.path.abspath(__file__))


def api(settings: config.Settings, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(settings.api_url + path,
                                 data=None if body is None else json.dumps(body).encode(),
                                 method="GET" if body is None else "POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"netinfermon service: {e.read().decode()}") from None
    except OSError:
        raise SystemExit(f"netinfermon service not reachable at {settings.api_url}. Start it with `netinfermon serve`.") from None


def run_dir(settings: config.Settings, spec: str) -> str:
    path = spec if os.path.isdir(spec) else os.path.join(settings.data_dir, spec)
    if not os.path.exists(os.path.join(path, "manifest.json")):
        raise SystemExit(f"no run at {path} (expected CAMPAIGN/LABEL under {settings.data_dir})")
    return path


# --- commands ----------------------------------------------------------------------------

def cmd_serve(s: config.Settings, a) -> None:
    from . import monitor
    svc = monitor.get_service()
    if svc.api_error:
        print(svc.api_error, file=sys.stderr)
    print(f"netinfermon {__version__}: devices {', '.join(d['name'] for d in svc.devices_info if d['active']) or 'none'}; "
          f"data in {s.data_dir}; page on http://{a.host or s.web_host}:{a.port or s.web_port}", flush=True)
    from streamlit.web import bootstrap
    flags = {"server_port": a.port or s.web_port, "server_address": a.host or s.web_host,
             "server_headless": True, "server_fileWatcherType": "none", "browser_gatherUsageStats": False}
    bootstrap.load_config_options(flag_options=flags)
    bootstrap.run(os.path.join(HERE, "web", "app.py"), False, [], flags)


def cmd_doctor(s: config.Settings, a) -> None:
    ok = True
    devs = sampler.discover(s.sysfs_root)
    print(f"RDMA devices under {s.sysfs_root}:")
    if not devs:
        print("  none found. Is this a host with a ConnectX / mlx5 NIC and the RDMA drivers loaded?")
        ok = False
    for d in devs:
        print(f"  {d['name']:16s} {d['state']:10s} {d['link_layer']:10s} {d['rate'] or '?':24s} "
              f"counters {'readable' if d['counters_readable'] else 'NOT readable'}")
    active = [d for d in devs if d["active"]]
    if devs and not active:
        print("  no ACTIVE device: the sampler would have nothing to read")
        ok = False
    nominal = sampler.nominal_gbps(devs)
    print(f"Link: nominal {nominal or '?'} Gb/s; configured maximum {s.link_max_gbps or 'not set'}, "
          f"one device {s.one_device_gbps or 'not set'} (config: {config.CONFIG_PATH})")
    try:
        path = sampler.binary()
        print(f"Sampler: {path}")
        if active:
            out = subprocess.run([path, "-n", "500", "-i", str(s.interval_us), "-b", s.sysfs_root],
                                 capture_output=True, text=True, timeout=30).stdout.splitlines()
            ticks = [int(l.split(",")[3]) for l in out if l.startswith("S,") and l.split(",")[2] == "0"]
            if len(ticks) > 2:
                per = (ticks[-1] - ticks[0]) / (len(ticks) - 1) / 1e6
                print(f"  500 ticks: {per:.3f} ms per tick (target {s.interval_us / 1000:g} ms), "
                      f"{sum(l.startswith('E,') for l in out)} read errors")
    except Exception as e:                                       # noqa: BLE001
        print(f"Sampler: could not build or run: {e}")
        ok = False
    for name, port, host in (("web page", s.web_port, "0.0.0.0"), ("control API", s.api_port, "127.0.0.1")):
        with socket.socket() as sock:
            busy = sock.connect_ex(("127.0.0.1", port)) == 0
        print(f"Port {port} ({name}): {'in use (another netinfermon running?)' if busy else 'free'}")
    print(f"Data directory: {s.data_dir}")
    print("OK" if ok else "Problems found (see above)")
    sys.exit(0 if ok else 1)


def cmd_plans(s, a) -> None:
    for name, desc in loadgen.list_plans().items():
        plan = loadgen.load_plan(name)
        pts = ", ".join(f"{p['id']}: c{p['concurrency']} {p['input_tokens']}→{p['output_tokens']}" for p in plan["points"])
        print(f"{name}\n  {desc}\n  {pts}\n  {plan['warmup_rounds']} warm-up + {plan['measured_rounds']} measured rounds\n")


def show_status(st: dict) -> str:
    p = st.get("progress", {})
    return (f"{st.get('campaign')}/{st.get('label')}: {st.get('state')} · {p.get('done', 0)}/{p.get('total', 0)} rounds"
            + (f" · {p['current']}" if p.get("current") else "") + (f" · {st['error']}" if st.get("error") else ""))


def cmd_run(s: config.Settings, a) -> None:
    plan = loadgen.load_plan(a.plan) if a.plan.endswith(".json") else a.plan
    st = api(s, "/run/start", {"campaign": a.campaign, "label": a.label, "plan": plan,
                               "endpoint": a.endpoint, "model": a.model, "expect": a.expect, "notes": a.notes})
    print(show_status(st))
    if a.wait:
        follow(s)


def follow(s: config.Settings) -> None:
    last = None
    while True:
        st = api(s, "/run/status")
        line = show_status(st)
        if line != last:
            print(f"{time.strftime('%H:%M:%S')} {line}", flush=True)
            last = line
        if st.get("state") not in ("starting", "waiting for idle link", "running"):
            return
        time.sleep(5)


def cmd_status(s, a) -> None:
    st = api(s, "/run/status")
    if st.get("state") == "none":
        print("no run since the service started")
        return
    print(show_status(st))
    print("\n".join(st.get("log", [])[-a.lines:]))
    if a.follow:
        follow(s)


def cmd_stop(s, a) -> None:
    print(show_status(api(s, "/run/stop", {})))


def cmd_record(s: config.Settings, a) -> None:
    if not a.command:
        raise SystemExit("give the command to run after --")
    rdir = os.path.join(s.data_dir, a.campaign, a.label)
    os.makedirs(os.path.join(rdir, "network"), exist_ok=True)
    n = len([f for f in os.listdir(os.path.join(rdir, "network")) if f.startswith("recording-")])
    path = api(s, "/record/start", {"path": os.path.join(rdir, "network", f"recording-{n + 1}.csv")})["path"]
    api(s, "/marker", {"label": "command: " + " ".join(a.command)[:200]})
    print(f"recording to {path}", flush=True)
    try:
        rc = subprocess.call(a.command)
    finally:
        api(s, "/marker", {"label": "command finished"})
        api(s, "/record/stop", {})
    print(f"recording stopped; per-marker summary: netinfermon summarize {path}")
    sys.exit(rc)


def cmd_summarize(s: config.Settings, a) -> None:
    import csv

    from .rates import Recording, summarize
    rec = Recording(a.recording, a.link)
    h = rec.state.health()
    print(f"Devices: {', '.join(rec.devices)} · samples {h['samples']} · read errors {h['errors']} "
          f"· counter resets {h['resets']}")
    if h["tick_p50_ms"] is not None:
        print(f"Tick: p50 {h['tick_p50_ms']:.2f} ms, p95 {h['tick_p95_ms']:.2f} ms · "
              f"read half-width p95 {h['halfw_p95_us']:.0f} µs")
    rows = summarize(rec, a.threshold, link_gbps=a.link, target_gbps=a.target)
    if not rows:
        raise SystemExit("nothing to summarize")
    cols = list(rows[0])
    fmt = lambda v: "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    widths = {c: max(len(c), *(len(fmt(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(fmt(r[c]).ljust(widths[c]) for c in cols))
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"CSV: {a.csv}")


def cmd_report(s: config.Settings, a) -> None:
    from . import analysis
    rdir = run_dir(s, a.run)
    res = analysis.report(rdir, a.charts, rounds="all" if a.all_rounds else "plan")
    with open(os.path.join(rdir, "report", "report.md"), encoding="utf-8") as fh:
        print(fh.read())
    print(f"written to {os.path.join(rdir, 'report')}")
    if res["warnings"]:
        sys.exit(1)


def cmd_compare(s: config.Settings, a) -> None:
    from . import compare
    base, test = run_dir(s, a.base), run_dir(s, a.test)
    out = os.path.join(os.path.dirname(base), f"compare_{os.path.basename(base)}_vs_{os.path.basename(test)}")
    res = compare.compare(base, test, out)
    print(compare.markdown(res))
    print(f"written to {out}")


def cmd_fetch(s: config.Settings, a) -> None:
    remote_dir = a.remote_dir.rstrip("/") + "/" + (a.campaign + "/" if a.campaign else "")
    local = os.path.join(s.data_dir, a.campaign or "")
    os.makedirs(local, exist_ok=True)
    cmd = ["rsync", "-a", "--info=stats1", "--exclude", "report/", f"{a.host}:{remote_dir}", local.rstrip("/") + "/"]
    print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    sys.exit(subprocess.call(cmd))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="netinfermon", description="Inference and RDMA traffic monitor")
    ap.add_argument("--version", action="version", version=f"netinfermon {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="start the service and the web page")
    p.add_argument("--port", type=int)
    p.add_argument("--host")
    sub.add_parser("doctor", help="check RDMA devices, counters, sampler and ports")
    sub.add_parser("plans", help="list built-in plans")

    p = sub.add_parser("run", help="start a benchmark run in the service")
    p.add_argument("campaign")
    p.add_argument("label")
    p.add_argument("--plan", default="linkspeed", help="built-in plan name or a JSON file")
    p.add_argument("--endpoint")
    p.add_argument("--model")
    p.add_argument("--expect", choices=["all", "one", "any"], default="any",
                   help="link check: traffic on every device / on one device / no check")
    p.add_argument("--notes")
    p.add_argument("--wait", action="store_true", help="follow the run until it ends")

    p = sub.add_parser("status", help="state of the current run")
    p.add_argument("-n", "--lines", type=int, default=10)
    p.add_argument("-f", "--follow", action="store_true")
    sub.add_parser("stop", help="stop the current run after its current round")

    p = sub.add_parser("record", help="record traffic while a command runs")
    p.add_argument("campaign")
    p.add_argument("label")
    p.add_argument("command", nargs=argparse.REMAINDER)

    p = sub.add_parser("summarize", help="per-marker summary of a recording file")
    p.add_argument("recording")
    p.add_argument("--link", type=float, default=200.0, help="link maximum, Gb/s per direction")
    p.add_argument("--target", type=float, default=100.0, help="link speed to compare against, Gb/s")
    p.add_argument("--threshold", type=float, default=0.5, help="5 ms rate counted as traffic, Gb/s")
    p.add_argument("--csv", help="also write the summary as CSV")

    p = sub.add_parser("report", help="build a run's report")
    p.add_argument("run", help="CAMPAIGN/LABEL or a run directory")
    p.add_argument("--charts", choices=["html", "html+svg"])
    p.add_argument("--all-rounds", action="store_true", help="also use rounds beyond the plan")

    p = sub.add_parser("compare", help="compare two runs (reports must exist)")
    p.add_argument("base")
    p.add_argument("test")

    p = sub.add_parser("fetch", help="copy runs from a node with rsync over ssh")
    p.add_argument("host", help="[user@]host, as for ssh")
    p.add_argument("campaign", nargs="?")
    p.add_argument("--remote-dir", default=".local/share/netinfermon")

    a = ap.parse_args(argv)
    if a.cmd == "record" and a.command[:1] == ["--"]:
        a.command = a.command[1:]
    s = config.load()
    {"serve": cmd_serve, "doctor": cmd_doctor, "plans": cmd_plans, "run": cmd_run, "status": cmd_status,
     "stop": cmd_stop, "record": cmd_record, "summarize": cmd_summarize, "report": cmd_report, "compare": cmd_compare,
     "fetch": cmd_fetch}[a.cmd](s, a)


if __name__ == "__main__":
    main()
