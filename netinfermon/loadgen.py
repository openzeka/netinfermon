"""loadgen.py — load generator: OpenAI-compatible streaming client and the plan runner.

The request code is adapted from openzeka/llm-benchmark
(https://github.com/openzeka/llm-benchmark, GPL-3.0): forced output length with
ignore_eos + min_tokens, no silent retries, a stalled stream is an error, prompts padded from
the tool's prompt set, input sizing on the server's tokenizer, a unique tag at the start of
every request so the prefix cache cannot serve it. Every timestamp is CLOCK_MONOTONIC_RAW,
the clock the RDMA sampler uses, so each request lines up with the traffic it caused.

A plan is a JSON file:

    {"points": [{"id": "P1", "concurrency": 1, "input_tokens": 1024, "output_tokens": 1024}, ...],
     "warmup_rounds": 1, "measured_rounds": 3, "seed": 20261013,
     "idle_gbps": 0.5, "idle_seconds": 10, "read_timeout_s": 900}

Every point runs warm-up + measured rounds; a round sends `concurrency` requests at once.
Every request is written to disk as soon as its round finishes, so a stopped run continues
where it stopped when started again with the same campaign and label.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import secrets
import ssl
import statistics
import threading
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import count

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS_FILE = os.path.join(HERE, "data", "prompts.txt")
PLANS_DIR = os.path.join(HERE, "plans")

API_KEY = "Dummy"
OUTPUT_HINT = "Your answer must contain at least 128 tokens."
FALLBACK_CHARS_PER_TOKEN = 4.0
TAG_CHARS = len("[000000-00000] ")
PLAN_DEFAULTS = {"warmup_rounds": 1, "measured_rounds": 3, "seed": 0, "idle_gbps": 0.5,
                 "idle_seconds": 10, "read_timeout_s": 900, "force_output_length": True}

REQUEST_FIELDS = ["label", "point", "round", "warmup", "slot", "concurrency", "input_target",
                  "prompt_chars", "prompt_tokens", "output_target", "output_tokens", "output_ok",
                  "ttft_ms", "itl_ms", "decode_tps", "e2e_tps", "latency_s",
                  "t0_ns", "t_first_ns", "t_last_ns", "t_end_ns", "error"]

log = logging.getLogger("netinfermon.run")


def mono_raw_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


# --- prompts ---------------------------------------------------------------------------

def load_prompts() -> list[str]:
    """The prompt set, in file order (the same prompts in every run of a campaign)."""
    with open(PROMPTS_FILE, encoding="utf-8") as f:
        return [p.strip() for p in f if p.strip()]


def set_output_hint(prompt: str, output_tokens: int) -> str:
    """Point the prompt's length instruction at the chosen output length."""
    return prompt.replace(OUTPUT_HINT, f"Your answer must contain at least {output_tokens} tokens.")


def pad_prompt(prompt: str, filler_pool: list[str], target_chars: int, rng: random.Random) -> str:
    """Extend a prompt to target_chars with background text drawn from the other prompts.

    The question stays first and the filler order is drawn fresh on every call, so requests
    do not share a long prefix that a server's prefix cache could serve without prefill."""
    if len(prompt) >= target_chars:
        return prompt
    parts = [prompt, "", "Background notes (for reference only):"]
    length = sum(len(p) + 1 for p in parts)
    while length < target_chars:
        for item in rng.sample(filler_pool, len(filler_pool)):
            parts.append(item)
            length += len(item) + 1
            if length >= target_chars:
                break
    return "\n".join(parts)[:target_chars]


# --- client ------------------------------------------------------------------------------

def get_models(endpoint: str, timeout: float = 30, verify_tls: bool = True) -> dict:
    ctx = None if verify_tls else ssl._create_unverified_context()
    with urllib.request.urlopen(endpoint.rstrip("/") + "/models", timeout=timeout, context=ctx) as r:
        return json.loads(r.read())


class Client:
    """Streaming chat-completions client. One instance serves concurrent requests."""

    def __init__(self, endpoint: str, model: str, force_output_length: bool = True,
                 read_timeout: float = 900.0, max_concurrency: int = 8, api_key: str = API_KEY,
                 verify_tls: bool = True):
        import httpx
        from openai import OpenAI

        self.model = model
        self.force_output_length = force_output_length
        # A stream with no chunk for this long is cut and counted as an error
        self.chunk_timeout = max(100.0, read_timeout / 2.0)
        self._http = httpx.Client(verify=verify_tls, limits=httpx.Limits(
            max_connections=max_concurrency + 32, max_keepalive_connections=max(20, max_concurrency)))
        # max_retries=0: a silently retried request would carry the failed attempt and the
        # retry back-off in its TTFT, so a failure is counted as an error instead
        self.client = OpenAI(base_url=endpoint.rstrip("/"), api_key=api_key or None,
                             timeout=httpx.Timeout(60.0, connect=10.0, read=read_timeout, write=10.0),
                             max_retries=0, http_client=self._http)

    def close(self) -> None:
        self._http.close()

    def measure(self, prompt: str, max_tokens: int, on_first=None) -> dict:
        """One streamed request. Times are CLOCK_MONOTONIC_RAW ns; on_first(t_ns) is called
        when the first token arrives."""
        import httpx

        t0 = mono_raw_ns()
        t_first = t_last = None
        chunks = tokens = prompt_tokens = 0
        try:
            extra = ({"extra_body": {"ignore_eos": True, "min_tokens": max_tokens}}
                     if self.force_output_length else {})
            stream = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, stream=True, stream_options={"include_usage": True}, **extra)
            last_chunk = time.monotonic()
            for event in stream:
                now_s = time.monotonic()
                if now_s - last_chunk > self.chunk_timeout:
                    return {"error": "chunk timeout: stream stalled", "t0_ns": t0, "t_end_ns": mono_raw_ns()}
                last_chunk = now_s
                if event.choices:
                    delta = event.choices[0].delta
                    content = (getattr(delta, "content", None) or getattr(delta, "reasoning_content", None)
                               or getattr(delta, "reasoning", None))
                    if content:
                        now = mono_raw_ns()
                        chunks += 1
                        if t_first is None:
                            t_first = now
                            if on_first:
                                on_first(now)
                        t_last = now
                if getattr(event, "usage", None) is not None:
                    tokens = event.usage.completion_tokens or 0
                    prompt_tokens = event.usage.prompt_tokens or 0
        except httpx.TimeoutException as e:
            return {"error": f"timeout: {type(e).__name__}", "t0_ns": t0, "t_end_ns": mono_raw_ns()}
        except Exception as e:                                       # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}", "t0_ns": t0, "t_end_ns": mono_raw_ns()}
        t_end = mono_raw_ns()
        if t_first is None:
            return {"error": "no content received", "t0_ns": t0, "t_end_ns": t_end}
        tokens = tokens or chunks
        itl_ms = (t_last - t_first) / 1e6 / (tokens - 1) if tokens > 1 else 0.0
        return {"t0_ns": t0, "t_first_ns": t_first, "t_last_ns": t_last, "t_end_ns": t_end,
                "ttft_ms": (t_first - t0) / 1e6, "itl_ms": itl_ms,
                "decode_tps": 1000.0 / itl_ms if itl_ms > 0 else None,
                "e2e_tps": tokens / ((t_end - t0) / 1e9), "latency_s": (t_end - t0) / 1e9,
                "tokens": tokens, "prompt_tokens": prompt_tokens}

    def size_input(self, prompt: str, filler_pool: list[str], input_tokens: int) -> int | None:
        """Prompt length in characters that gives input_tokens prompt tokens on this server
        (its tokenizer and chat template): a straight-line fit of usage.prompt_tokens against
        characters from two short requests. None if the server reports no prompt_tokens."""
        guess = max(int(input_tokens * FALLBACK_CHARS_PER_TOKEN), 2 * len(prompt))
        padded = pad_prompt(prompt, filler_pool, guess, random.Random(0))
        short = self.measure(prompt, 16).get("prompt_tokens", 0)
        long = self.measure(padded, 16).get("prompt_tokens", 0)
        if not short or long <= short:
            return None
        chars_per_token = (len(padded) - len(prompt)) / (long - short)
        return int(len(prompt) + (input_tokens - short) * chars_per_token)


# --- plans -------------------------------------------------------------------------------

def list_plans() -> dict[str, str]:
    """Built-in plans: name -> description."""
    out = {}
    for fn in sorted(os.listdir(PLANS_DIR)):
        if fn.endswith(".json"):
            with open(os.path.join(PLANS_DIR, fn), encoding="utf-8") as f:
                out[fn[:-5]] = json.load(f).get("description", "")
    return out


def load_plan(name_or_path: str) -> dict:
    path = name_or_path if os.path.exists(name_or_path) else os.path.join(PLANS_DIR, f"{name_or_path}.json")
    with open(path, encoding="utf-8") as f:
        plan = json.load(f)
    return {**PLAN_DEFAULTS, **plan}


def build_plan(input_tokens: list[int], output_tokens: int, concurrency: list[int],
               warmup_rounds: int = 1, measured_rounds: int = 3, seed: int = 0,
               force_output_length: bool = True) -> dict:
    """A plan from simple settings: every input length x every concurrency level."""
    points = [{"id": f"c{c}-in{n}", "concurrency": c, "input_tokens": n, "output_tokens": output_tokens}
              for c in concurrency for n in input_tokens]
    return {**PLAN_DEFAULTS, "description": "custom", "points": points, "warmup_rounds": warmup_rounds,
            "measured_rounds": measured_rounds, "seed": seed, "force_output_length": force_output_length}


def check_link(expect: str, before: dict[str, int], after: dict[str, int]) -> str | None:
    """Error text if the per-device traffic split does not match `expect`, "ok:{...}" if it
    does, None if there is too little traffic to judge yet.

    expect: "all" = every device carries traffic (each at least 20%/n of it); "one" = one
    device carries at least 99%; "any" = no check."""
    delta = {d: after[d] - before.get(d, after[d]) for d in after}
    total = sum(delta.values())
    if total < 10_000_000:
        return None
    shares = {d: v / total for d, v in delta.items()}
    if expect == "all" and min(shares.values()) < 0.2 / len(shares):
        return f"expected traffic on every device, got shares {shares}"
    if expect == "one" and max(shares.values()) < 0.99:
        return f"expected traffic on one device only, got shares {shares}"
    return "ok:" + json.dumps({d: round(s, 4) for d, s in shares.items()})


# --- files -------------------------------------------------------------------------------

def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_requests(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_requests(path: str, rows: list[dict]) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REQUEST_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


# --- runner ------------------------------------------------------------------------------

class Abort(Exception):
    pass


class Runner:
    """Runs one plan for one campaign/label against an endpoint, recording the traffic through
    `monitor` (the monitor service, in the same process). State is readable while it runs."""

    def __init__(self, monitor, data_dir: str, campaign: str, label: str, plan: dict,
                 endpoint: str, model: str | None = None, expect: str = "any",
                 link_max_gbps: float | None = None, notes: str | None = None,
                 verify_tls: bool = True):
        self.monitor = monitor
        self.verify_tls = verify_tls
        self.data_dir = data_dir
        self.campaign, self.label, self.plan = campaign, label, {**PLAN_DEFAULTS, **plan}
        self.endpoint, self.model, self.expect = endpoint, model, expect
        self.link_max_gbps, self.notes = link_max_gbps, notes
        self.dir = os.path.join(data_dir, campaign, label)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.state = "starting"
        self.error: str | None = None
        self.started = datetime.now()
        self.progress = {"done": 0, "total": 0, "current": ""}
        self.recent: deque = deque(maxlen=200)       # finished request rows, for the page
        self.inflight: dict[int, dict] = {}          # request id -> {t0, t_first} (mono raw ns)
        self.log_lines: deque = deque(maxlen=300)
        self._ids = count()

    # --- status for the page and the API ---

    def status(self) -> dict:
        with self.lock:
            return {"state": self.state, "error": self.error, "campaign": self.campaign,
                    "label": self.label, "endpoint": self.endpoint, "model": self.model,
                    "expect": self.expect, "started": self.started.isoformat(timespec="seconds"),
                    "progress": dict(self.progress), "dir": self.dir,
                    "log": list(self.log_lines)[-20:]}

    def timeline(self) -> tuple[list[dict], list[dict]]:
        """(finished, in flight) requests with their mono-raw times, for the live chart."""
        with self.lock:
            return list(self.recent), [dict(v) for v in self.inflight.values()]

    def _log(self, level: int, msg: str, *args) -> None:
        text = msg % args if args else msg
        log.log(level, text)
        line = f"{datetime.now():%H:%M:%S} {logging.getLevelName(level)} {text}"
        with self.lock:
            self.log_lines.append(line)
        try:
            with open(os.path.join(self.dir, "run.log"), "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    # --- run ---

    def run(self) -> None:
        try:
            self._run()
            with self.lock:
                self.state = "stopped" if self.stop_event.is_set() else "done"
        except Abort as e:
            self._log(logging.ERROR, "ABORTED: %s", e)
            with self.lock:
                self.state, self.error = "aborted", str(e)
        except Exception as e:                                       # noqa: BLE001
            self._log(logging.ERROR, "FAILED: %s: %s", type(e).__name__, e)
            with self.lock:
                self.state, self.error = "failed", f"{type(e).__name__}: {e}"

    def _run(self) -> None:
        plan = self.plan
        os.makedirs(os.path.join(self.dir, "network"), exist_ok=True)
        req_path = os.path.join(self.dir, "requests.csv")
        man_path = os.path.join(self.dir, "manifest.json")
        manifest = read_json(man_path) if os.path.exists(man_path) else {}
        if manifest and manifest.get("expect") != self.expect:
            raise Abort(f"{self.label} was started with expect={manifest.get('expect')}, not {self.expect}")

        models = get_models(self.endpoint, verify_tls=self.verify_tls)
        ids = [m["id"] for m in models.get("data", [])]
        model = self.model or (ids[0] if len(ids) == 1 else None)
        if model not in ids:
            raise Abort(f"model {self.model!r} not served at {self.endpoint}; available: {ids}")
        self.model = model
        need_ctx = max(p["input_tokens"] + p["output_tokens"] for p in plan["points"]) + 256
        max_len = next((m.get("max_model_len") for m in models["data"] if m["id"] == model), None)
        if max_len and max_len < need_ctx:
            raise Abort(f"model context {max_len} is smaller than the plan needs ({need_ctx})")

        st = self.monitor.status()
        if not st["alive"]:
            raise Abort("the RDMA sampler is not running")
        if st["recording"]:
            raise Abort(f"the monitor is already recording ({st['rec_path']})")

        manifest.setdefault("campaign", self.campaign)
        manifest.setdefault("label", self.label)
        manifest.setdefault("expect", self.expect)
        manifest.setdefault("created", datetime.now().isoformat(timespec="seconds"))
        manifest.update({"model": model, "endpoint": self.endpoint, "plan": plan,
                         "host": os.uname().nodename, "devices": st["devices"],
                         "link_max_gbps": self.link_max_gbps, "notes": self.notes or manifest.get("notes")})
        manifest.setdefault("runs", []).append({"started": datetime.now().isoformat(timespec="seconds")})
        write_json(man_path, manifest)
        write_json(os.path.join(self.dir, "models.json"), models)

        sizing_path = os.path.join(self.data_dir, self.campaign, "sizing.json")
        sizing = read_json(sizing_path) if os.path.exists(sizing_path) else {}
        done = {(r["point"], int(r["round"])) for r in read_requests(req_path)}
        n_rounds = plan["warmup_rounds"] + plan["measured_rounds"]
        todo = [(p, r) for p in plan["points"] for r in range(n_rounds) if (p["id"], r) not in done]
        with self.lock:
            self.progress = {"done": len(plan["points"]) * n_rounds - len(todo),
                             "total": len(plan["points"]) * n_rounds, "current": ""}
        if not todo:
            self._log(logging.INFO, "nothing to run: every round is recorded")
            return

        prompts = load_prompts()
        hinted_by_out: dict[int, list[str]] = {}
        filler = [p.replace(OUTPUT_HINT, "").strip() for p in prompts]
        client = Client(self.endpoint, model, plan["force_output_length"], plan["read_timeout_s"],
                        max(p["concurrency"] for p in plan["points"]), verify_tls=self.verify_tls)
        run_id = secrets.token_hex(3)
        req_no = count()
        manifest["runs"][-1]["run_id"] = run_id
        write_json(man_path, manifest)
        with self.lock:
            self.state = "waiting for idle link"
        self._log(logging.INFO, "waiting for an idle link (%s Gb/s for %s s)", plan["idle_gbps"], plan["idle_seconds"])
        self._wait_idle()
        n_rec = len([f for f in os.listdir(os.path.join(self.dir, "network")) if f.startswith("recording-")])
        rec_path = self.monitor.start_recording(os.path.join(self.dir, "network", f"recording-{n_rec + 1}.csv"),
                                                owner="run")
        self._log(logging.INFO, "recording to %s", rec_path)
        with self.lock:
            self.state = "running"
        link_checked = manifest.get("link_check", "").startswith("ok:")
        try:
            for point, rnd in todo:
                if self.stop_event.is_set():
                    self._log(logging.INFO, "stopped by request; rerun the same label to continue")
                    break
                pid, conc = point["id"], point["concurrency"]
                n_in, n_out = point["input_tokens"], point["output_tokens"]
                hinted = hinted_by_out.setdefault(n_out, [set_output_hint(p, n_out) for p in prompts])
                key = f"{model}|{n_in}|{n_out}"
                if key not in sizing:
                    self.monitor.marker(f"{pid} sizing")
                    sizing[key] = client.size_input(hinted[0], filler, n_in)
                    os.makedirs(os.path.dirname(sizing_path), exist_ok=True)
                    write_json(sizing_path, sizing)
                target_chars = sizing[key] or int(n_in * FALLBACK_CHARS_PER_TOKEN)

                rng = random.Random(f"{plan['seed']}:{pid}:{rnd}")
                texts = [f"[{run_id}-{next(req_no):05d}] "
                         + pad_prompt(hinted[i], filler, target_chars - TAG_CHARS, rng)
                         for i in rng.sample(range(len(hinted)), conc)]
                warm = rnd < plan["warmup_rounds"]
                label = (f"{pid} c{conc} in{n_in} out{n_out} round {rnd + 1}/{n_rounds} "
                         f"{'warm-up' if warm else 'measured'}")
                self.monitor.marker(label)
                self._log(logging.INFO, label)
                with self.lock:
                    self.progress["current"] = label

                before = self._tx_bytes()
                barrier = threading.Barrier(conc)

                def one(slot: int, text: str):
                    rid = next(self._ids)
                    barrier.wait()
                    with self.lock:
                        self.inflight[rid] = {"t0": mono_raw_ns(), "t_first": None, "label": label}

                    def first(t):
                        with self.lock:
                            if rid in self.inflight:
                                self.inflight[rid]["t_first"] = t
                    res = client.measure(text, n_out, on_first=first)
                    with self.lock:
                        self.inflight.pop(rid, None)
                    return slot, res

                with ThreadPoolExecutor(max_workers=conc) as ex:
                    outs = list(ex.map(lambda a: one(*a), enumerate(texts)))
                after = self._tx_bytes()

                rows = []
                for slot, res in sorted(outs, key=lambda o: o[0]):
                    row = {"label": self.label, "point": pid, "round": rnd, "warmup": int(warm), "slot": slot,
                           "concurrency": conc, "input_target": n_in, "prompt_chars": len(texts[slot]),
                           "output_target": n_out, "t0_ns": res["t0_ns"], "t_end_ns": res["t_end_ns"],
                           "error": res.get("error", "")}
                    if "error" not in res:
                        row.update({"prompt_tokens": res["prompt_tokens"], "output_tokens": res["tokens"],
                                    "output_ok": int(res["tokens"] == n_out),
                                    "ttft_ms": round(res["ttft_ms"], 3), "itl_ms": round(res["itl_ms"], 4),
                                    "decode_tps": "" if res["decode_tps"] is None else round(res["decode_tps"], 3),
                                    "e2e_tps": round(res["e2e_tps"], 3), "latency_s": round(res["latency_s"], 4),
                                    "t_first_ns": res["t_first_ns"], "t_last_ns": res["t_last_ns"]})
                        if res["tokens"] != n_out and plan["force_output_length"]:
                            self._log(logging.WARNING, "%s slot %s: %s output tokens instead of %s",
                                      label, slot, res["tokens"], n_out)
                    else:
                        self._log(logging.ERROR, "%s slot %s: %s", label, slot, res["error"])
                    rows.append(row)
                append_requests(req_path, rows)
                with self.lock:
                    self.recent.extend(rows)
                    self.progress["done"] += 1

                if not link_checked and self.expect != "any":
                    verdict = check_link(self.expect, before, after)
                    if verdict is None:
                        pass
                    elif verdict.startswith("ok:"):
                        manifest["link_check"] = verdict
                        write_json(man_path, manifest)
                        link_checked = True
                        self._log(logging.INFO, "link check (%s): %s", self.expect, verdict)
                    else:
                        raise Abort(f"link check failed for expect={self.expect}: {verdict}")
        finally:
            self.monitor.stop_recording()
            client.close()
            manifest["runs"][-1]["ended"] = datetime.now().isoformat(timespec="seconds")
            write_json(man_path, manifest)
        self._log(logging.INFO, "raw data complete for %s/%s", self.campaign, self.label)

    def _tx_bytes(self) -> dict[str, int]:
        return {d: c["tx_bytes"] for d, c in self.monitor.status()["counters"].items()}

    def _wait_idle(self) -> None:
        need, limit = self.plan["idle_seconds"], self.plan["idle_gbps"]
        deadline = time.time() + 120
        quiet_since = None
        while time.time() < deadline:
            if self.stop_event.is_set():
                raise Abort("stopped while waiting for an idle link")
            now = self.monitor.status()["now"]
            rates = [now.get("tx_100ms_gbps"), now.get("rx_100ms_gbps")]
            if all(r is not None and r < limit for r in rates):
                quiet_since = quiet_since or time.time()
                if time.time() - quiet_since >= need:
                    return
            else:
                quiet_since = None
            time.sleep(1)
        raise Abort(f"link not idle: traffic above {limit} Gb/s for 2 minutes. Is another workload using the network?")


def request_summary(rows: list[dict]) -> list[dict]:
    """Per point, measured rounds only: request count, errors, TTFT / decode / end-to-end
    means. Cheap (no traffic data), shown while a run is going and right after it."""
    out = {}
    for r in rows:
        if r["warmup"] in ("1", 1):
            continue
        p = out.setdefault(r["point"], {"point": r["point"], "concurrency": int(r["concurrency"]),
                                        "input_tokens": int(r["input_target"]),
                                        "output_tokens": int(r["output_target"]),
                                        "requests": 0, "errors": 0, "_ttft": [], "_dec": [], "_e2e": [],
                                        "_lat": [], "_ptok": []})
        p["requests"] += 1
        if r["error"]:
            p["errors"] += 1
            continue
        p["_ttft"].append(float(r["ttft_ms"]))
        if r.get("decode_tps"):
            p["_dec"].append(float(r["decode_tps"]))
        p["_e2e"].append(float(r["e2e_tps"]))
        p["_lat"].append(float(r["latency_s"]))
        if r.get("prompt_tokens"):
            p["_ptok"].append(int(r["prompt_tokens"]))
    res = []
    for p in out.values():
        m = lambda xs: statistics.mean(xs) if xs else None
        res.append({k: v for k, v in p.items() if not k.startswith("_")} | {
            "ttft_ms_mean": m(p["_ttft"]), "decode_tps_mean": m(p["_dec"]), "e2e_tps_mean": m(p["_e2e"]),
            "latency_s_mean": m(p["_lat"]), "prompt_tokens_mean": m(p["_ptok"])})
    return res
