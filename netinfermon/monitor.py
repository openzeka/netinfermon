"""monitor.py — the netinfermon service: RDMA sampler, live state, recordings, benchmark runs and
a local control API.

One instance per process. `netinfermon serve` starts it before the web page, so sampling and runs
do not depend on a browser. The control API listens on 127.0.0.1 only (default port 8599):

    GET  /status          sampler / recording state, per-device byte counters, current rates
    POST /record/start    {"path": optional absolute path under $HOME}  -> {"path": ...}
    POST /marker          {"label": "..."}                              -> {"t_ns": ...}
    POST /record/stop     {}                                            -> {"path": ...}
    POST /run/start       {"campaign", "label", "plan" (name, path or dict), "endpoint",
                           "model", "expect": "all|one|any", "notes"}   -> run status
    GET  /run/status      status of the current (or last) run
    POST /run/stop        stop after the current round                  -> run status

All times are CLOCK_MONOTONIC_RAW nanoseconds, the clock the sampler uses.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config, sampler
from .loadgen import Runner, load_plan
from .rates import BYTES_PER_WORD, DIRS, TOTAL, WINDOWS_MS, LiveState

CHART_STEP_NS = 50_000_000      # one chart point per 50 ms of sample time
CHART_KEEP = 12000              # 12000 x 50 ms = 10 minutes of history


def mono_raw_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


class Service:
    """Runs the sampler process; feeds its lines to LiveState and to a recording, if any."""

    def __init__(self, settings: config.Settings | None = None):
        self.settings = settings or config.load()
        s = self.settings
        self.devices_info = sampler.discover(s.sysfs_root)
        self.nominal_gbps = sampler.nominal_gbps(self.devices_info)
        self.link_gbps = s.link_max_gbps or self.nominal_gbps or 200.0
        self.lock = threading.Lock()
        self.state = LiveState(link_gbps=self.link_gbps,
                               device_limit_gbps=self.nominal_gbps * 1.05 if self.nominal_gbps else None)
        self.chart: deque = deque(maxlen=CHART_KEEP)
        self.header: dict[str, str] = {}            # "H", "D<idx>", "C" -> latest line
        self.stderr: deque = deque(maxlen=20)
        self.rec_file = None
        self.rec_path: str | None = None
        self.rec_started = 0.0
        self.rec_owner = "ui"                       # "ui", "api" or "run"
        self.proc: subprocess.Popen | None = None
        self.api: ThreadingHTTPServer | None = None
        self.api_error: str | None = None
        self.runner: Runner | None = None
        self.run_thread: threading.Thread | None = None
        self._next_chart = 0

    def start(self) -> None:
        s = self.settings
        args = [sampler.binary(), "-i", str(s.interval_us), "-b", s.sysfs_root]
        if s.devices:
            args += ["-d", ",".join(s.devices)]
        self.proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, bufsize=1)
        threading.Thread(target=self._read_out, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()
        self._start_api()

    def stop(self) -> None:
        if self.runner:
            self.runner.stop_event.set()
        self.stop_recording()
        if self.api:
            self.api.shutdown()
            self.api.server_close()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _read_err(self) -> None:
        for line in self.proc.stderr:
            self.stderr.append(line.rstrip())

    def _read_out(self) -> None:
        for line in self.proc.stdout:
            kind = line[:1]
            with self.lock:
                if kind in "HC":
                    self.header[kind] = line
                elif kind == "D":
                    self.header["D" + line.split(",")[1]] = line
                self.state.feed(line)
                if self.rec_file:
                    self.rec_file.write(line)
                if kind == "S":
                    self._chart_point()

    def _chart_point(self) -> None:
        t = self.state.last_t()
        if t < self._next_chart:
            return
        self._next_chart = t + CHART_STEP_NS
        pt = {"t_ns": t, "t": datetime.fromtimestamp(self.state.wall(t))}
        for d in DIRS:
            for w in (WINDOWS_MS[0], 100):
                r = self.state.current(TOTAL, d, w)
                pt[f"{d}{w}"] = r.gbps if r else None
        self.chart.append(pt)

    # --- recording -----------------------------------------------------------------

    def start_recording(self, path: str | None = None, owner: str = "ui") -> str:
        with self.lock:
            if self.rec_file:
                raise RuntimeError(f"already recording: {self.rec_path}")
            if path is None:
                os.makedirs(self.settings.recordings_dir, exist_ok=True)
                path = os.path.join(self.settings.recordings_dir, f"rec-{datetime.now():%Y%m%d-%H%M%S}.csv")
            else:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            f = open(path, "x", encoding="utf-8")          # never overwrites a file
            for key in sorted(self.header, key=lambda k: (k[0] != "H", k[0] == "C", k)):
                f.write(self.header[key])
            f.write(f"M,{mono_raw_ns()},recording started\n")
            self.rec_file, self.rec_path, self.rec_started = f, path, time.time()
            self.rec_owner = owner
            self.state.restart("overall", "recording")
            self.state.restart("phase", "recording started")
            return path

    def marker(self, label: str) -> int:
        label = " ".join(label.replace(",", " ").split()) or "marker"
        with self.lock:
            t = mono_raw_ns()
            if self.rec_file:
                self.rec_file.write(f"M,{t},{label}\n")
                self.rec_file.flush()
                self.state.restart("phase", label)
            return t

    def stop_recording(self) -> str | None:
        with self.lock:
            if not self.rec_file:
                return None
            self.rec_file.write(f"M,{mono_raw_ns()},recording stopped\n")
            self.rec_file.close()
            self.rec_file = None
            self.state.freeze("overall")           # keep the last values on screen
            self.state.freeze("phase")
            return self.rec_path

    # --- benchmark runs ------------------------------------------------------------

    def link_max_for(self, expect: str) -> float | None:
        s = self.settings
        if expect == "one":
            return s.one_device_gbps
        return s.link_max_gbps or self.nominal_gbps

    def start_run(self, campaign: str, label: str, plan, endpoint: str | None = None,
                  model: str | None = None, expect: str = "any", notes: str | None = None) -> dict:
        if self.run_thread and self.run_thread.is_alive():
            raise RuntimeError(f"a run is already going: {self.runner.campaign}/{self.runner.label}")
        if expect not in ("all", "one", "any"):
            raise ValueError("expect must be all, one or any")
        for name, v in (("campaign", campaign), ("label", label)):
            if not v or os.sep in v or v.startswith(".") or v.startswith("_"):
                raise ValueError(f"invalid {name}: {v!r}")
        plan = load_plan(plan) if isinstance(plan, str) else plan
        self.runner = Runner(self, self.settings.data_dir, campaign, label, plan,
                             endpoint or self.settings.endpoint, model, expect,
                             self.link_max_for(expect), notes, verify_tls=not self.settings.insecure_tls)
        os.makedirs(self.runner.dir, exist_ok=True)
        self.run_thread = threading.Thread(target=self.runner.run, daemon=True, name="netinfermon-run")
        self.run_thread.start()
        return self.runner.status()

    def run_status(self) -> dict | None:
        return self.runner.status() if self.runner else None

    def stop_run(self) -> dict | None:
        if self.runner:
            self.runner.stop_event.set()
        return self.run_status()

    # --- control API (127.0.0.1 only) ----------------------------------------------

    def status(self) -> dict:
        with self.lock:
            st = self.state
            counters = {}
            for dev in st.devices:
                ser_tx, ser_rx = st.series[(dev, "tx")], st.series[(dev, "rx")]
                if len(ser_tx) and len(ser_rx):
                    counters[dev] = {"t_ns": ser_tx.t[-1],
                                     "tx_bytes": ser_tx.v[-1] * BYTES_PER_WORD,
                                     "rx_bytes": ser_rx.v[-1] * BYTES_PER_WORD}
            now = {}
            for d in DIRS:
                for w in WINDOWS_MS:
                    r = st.current(TOTAL, d, w)
                    now[f"{d}_{w}ms_gbps"] = None if r is None else r.gbps
            return {"alive": self.alive, "recording": bool(self.rec_file), "rec_path": self.rec_path,
                    "rec_owner": self.rec_owner if self.rec_file else None,
                    "interval_us": self.settings.interval_us, "devices": list(st.devices),
                    "counters": counters, "now": now, "health": st.health(), "mono_raw_ns": mono_raw_ns(),
                    "link_gbps": self.link_gbps, "nominal_gbps": self.nominal_gbps,
                    "stderr": list(self.stderr)}

    def _start_api(self) -> None:
        svc = self
        home = os.path.expanduser("~") + os.sep

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code: int, obj) -> None:
                body = json.dumps(obj, default=str).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/status":
                    self._send(200, svc.status())
                elif self.path == "/run/status":
                    self._send(200, svc.run_status() or {"state": "none"})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    req = json.loads(self.rfile.read(n) or b"{}")
                    if self.path == "/record/start":
                        path = req.get("path")
                        if path is not None:
                            path = os.path.realpath(path)
                            if not path.startswith(home):
                                return self._send(400, {"error": "path must be under $HOME"})
                        self._send(200, {"path": svc.start_recording(path, owner="api")})
                    elif self.path == "/marker":
                        self._send(200, {"t_ns": svc.marker(str(req.get("label", "")))})
                    elif self.path == "/record/stop":
                        self._send(200, {"path": svc.stop_recording()})
                    elif self.path == "/run/start":
                        self._send(200, svc.start_run(req["campaign"], req["label"], req["plan"],
                                                      req.get("endpoint"), req.get("model"),
                                                      req.get("expect", "any"), req.get("notes")))
                    elif self.path == "/run/stop":
                        self._send(200, svc.stop_run() or {"state": "none"})
                    else:
                        self._send(404, {"error": "not found"})
                except (RuntimeError, FileExistsError) as e:
                    self._send(409, {"error": str(e)})
                except (ValueError, KeyError, FileNotFoundError) as e:
                    self._send(400, {"error": f"{type(e).__name__}: {e}"})
                except Exception as e:                      # noqa: BLE001
                    self._send(500, {"error": f"{type(e).__name__}: {e}"})

        try:
            self.api = ThreadingHTTPServer(("127.0.0.1", self.settings.api_port), Handler)
        except OSError as e:
            self.api_error = f"control API not started on 127.0.0.1:{self.settings.api_port}: {e}"
            return
        threading.Thread(target=self.api.serve_forever, daemon=True).start()


_service: Service | None = None
_service_lock = threading.Lock()


def get_service() -> Service:
    global _service
    with _service_lock:
        if _service is None:
            _service = Service()
            _service.start()
        return _service


def restart_service() -> Service:
    global _service
    with _service_lock:
        if _service is not None:
            _service.stop()
        _service = Service()
        _service.start()
        return _service
