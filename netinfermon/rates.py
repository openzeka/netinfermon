"""rates.py — rate maths for RDMA byte counters (standard library only).

The live monitor, the recording summary, the run analysis and the tests all use this code.

Definitions
-----------
* Counter values are 4-byte words: bytes = words x 4, bits = words x 32. bits / ns = Gb/s.
* A sample (t, v, h): t = midpoint of the counter read (ns, CLOCK_MONOTONIC_RAW),
  h = half-width (ns). The counter was really read at some moment in [t - h, t + h].
* Window W ending at sample i_end: j is the LAST sample with t_j <= t_end - W.
  Rate = (v_end - v_j) / (t_end - t_j). That is the EXACT mean over [t_j, t_end]; the real
  span is between W and W + one tick and is returned with the result. Unchanged samples are
  kept too, so idle time is counted correctly.
* Bounds: the real read moments are uncertain by +-h, so the real span lies in
  span +- (h_j + h_end). lo / hi are the exact lower / upper rate bounds that follow (counter
  values are exact; time is the only uncertainty).
* Coverage: if there is no sample before the window start, or the span exceeds 2W (a sampler
  gap), no value is produced. That avoids fake peaks from a short span at start-up or after
  a gap.
"""

from __future__ import annotations

import math
from array import array
from bisect import bisect_right
from collections import deque
from typing import NamedTuple

BITS_PER_WORD = 32
BYTES_PER_WORD = 4
WINDOWS_MS = (5, 100, 1000)
DIRS = ("tx", "rx")
TOTAL = "TOTAL"
ACTIVE_GBPS = 0.5          # a 5 ms window faster than this counts as "traffic"
# Rate histogram bins, as fractions of the link maximum (the first edge is absolute: 1 Gb/s)
HIST_FRACTIONS = (0.25, 0.50, 0.75)
HIST_LABELS = ("<1 Gb/s", "1 Gb/s–25%", "25–50%", "50–75%", "≥75%")


def _bin(gbps: float, link_gbps: float) -> int:
    edges = (1.0, *(f * link_gbps for f in HIST_FRACTIONS))
    for i, edge in enumerate(edges):
        if gbps < edge:
            return i
    return len(edges)


def link_estimate(gb: float, active_s: float | None, link_gbps: float,
                  target_gbps: float) -> dict:
    """Link bottleneck estimate.

    busy time = bits moved / the link's maximum rate (time the data flows at full speed)
    f         = busy time / active span (first to last 5 ms window with traffic)
    factor    = 1 - f + f x (link / target): how many times longer the active span would be if
                the link dropped from link_gbps to target_gbps. An upper bound: it assumes
                transfers run at full speed and do not overlap with compute (the rate
                distribution checks the first). It understates bursts shorter than the 5 ms
                window, so prefer the per-sample view for short collectives."""
    busy_s = gb * 8 / link_gbps
    f = busy_s / active_s if active_s else None
    return {"busy_s": busy_s, "active_s": active_s, "f": f,
            "factor": None if f is None else 1 - f + f * link_gbps / target_gbps}


class Rate(NamedTuple):
    gbps: float
    lo: float
    hi: float
    span_ns: int
    t_end: int

    @property
    def pm(self) -> float:
        """Largest deviation, for a symmetric +- display."""
        return max(self.hi - self.gbps, self.gbps - self.lo)


def sum_rates(rates: list[Rate]) -> Rate:
    return Rate(sum(r.gbps for r in rates), sum(r.lo for r in rates),
                sum(r.hi for r in rates), max(r.span_ns for r in rates),
                max(r.t_end for r in rates))


class Series:
    """Samples of one counter (device x direction)."""

    __slots__ = ("t", "v", "h", "keep_ns", "resets")

    def __init__(self, keep_s: float | None = 180.0):
        self.t = array("q")
        self.v = array("q")
        self.h = array("q")
        self.keep_ns = None if keep_s is None else int(keep_s * 1e9)
        self.resets = 0

    def __len__(self) -> int:
        return len(self.t)

    def add(self, t: int, v: int, h: int) -> None:
        if self.t:
            if v < self.v[-1]:          # counter went back: no rate across the old samples
                self.resets += 1
                del self.t[:], self.v[:], self.h[:]
            elif t <= self.t[-1]:       # time did not move: ignore the sample
                return
        self.t.append(t)
        self.v.append(v)
        self.h.append(h)
        if (self.keep_ns is not None and len(self.t) > 8192
                and t - self.t[0] > self.keep_ns * 5 // 4):
            cut = bisect_right(self.t, t - self.keep_ns)
            del self.t[:cut], self.v[:cut], self.h[:cut]

    def index_at(self, t: int) -> int:
        """Index of the last sample at or before t (-1 if none)."""
        return bisect_right(self.t, t) - 1

    def window(self, i_end: int, w_ns: int) -> Rate | None:
        """Rate of the W window ending at sample i_end; None if not covered."""
        if i_end < 0:
            return None
        t_end = self.t[i_end]
        j = bisect_right(self.t, t_end - w_ns, 0, i_end) - 1
        if j < 0:
            return None
        span = t_end - self.t[j]
        if span > 2 * w_ns:
            return None
        bits = (self.v[i_end] - self.v[j]) * BITS_PER_WORD
        u = self.h[i_end] + self.h[j]
        lo = bits / (span + u)
        hi = bits / (span - u) if span > u else math.inf
        return Rate(bits / span, lo, hi, span, t_end)

    def now(self, w_ns: int) -> Rate | None:
        return self.window(len(self.t) - 1, w_ns)


# ----------------------------------------------------------------------------
# Live state: sampler lines -> current windows, peaks, health
# ----------------------------------------------------------------------------

class Tracker:
    """Exact mean and window peaks since a starting moment (start-up, reset, recording start,
    marker). Mean = bits moved since the start / time elapsed; peak = the highest window rate
    in that time (sliding, every tick)."""

    def __init__(self, label: str):
        self.label = label
        self.base: dict[tuple[str, str], tuple[int, int]] = {}   # (device, dir) -> (t, v)
        self.end: dict[tuple[str, str], tuple[int, int]] | None = None   # when frozen
        self.peaks: dict[tuple[str, str, int], tuple[Rate, str]] = {}   # -> (rate, phase)
        # Link activity: 5 ms rate of max(TX, RX) over the whole port, one value per tick
        self.first_act: int | None = None      # start of the first window with traffic (ns)
        self.last_act: int | None = None       # end of the last window with traffic (ns)
        self.hist = [0] * len(HIST_LABELS)     # ticks per bin since the first traffic
        self.hist_last = [0] * len(HIST_LABELS)   # the same, as of the last tick with traffic

    @property
    def frozen(self) -> bool:
        return self.end is not None


class LiveState:
    """Feeds on sampler lines. link_gbps is the measured link maximum (per direction, all
    devices together); device_limit_gbps, if known, flags impossible per-device rates."""

    def __init__(self, windows_ms=WINDOWS_MS, keep_s: float | None = 180.0,
                 track_peaks: bool = True, link_gbps: float = 200.0,
                 device_limit_gbps: float | None = None):
        self.windows_ms = tuple(windows_ms)
        self.track_peaks = track_peaks
        self.link_gbps = link_gbps
        self.device_limit_gbps = device_limit_gbps
        self.devices: list[str] = []
        self.series: dict[tuple[str, str], Series] = {}
        self.keep_s = keep_s
        self.interval_us: int | None = None
        self.anchor: tuple[int, int] | None = None     # (mono_raw_ns, realtime_ns)
        self.tick: int | None = None
        self.tick_t0: deque = deque(maxlen=5000)       # first read moment of each tick
        self.halfw: deque = deque(maxlen=20000)        # read half-widths (ns)
        self.errors = 0
        self.impossible = 0
        self.samples = 0
        # "overall": since start-up / reset / recording start; "phase": since the last marker
        self.trackers: dict[str, Tracker] = {"overall": Tracker("since start"),
                                             "phase": Tracker("since start")}

    # --- input ------------------------------------------------------------------

    def feed(self, line: str) -> None:
        p = line.rstrip("\n").split(",")
        kind = p[0]
        try:
            if kind == "S" and len(p) == 9:
                self._new_tick(int(p[1]), int(p[3]))
                dev = self.devices[int(p[2])]
                for d, (a, b, v) in (("tx", p[3:6]), ("rx", p[6:9])):
                    a, b, v = int(a), int(b), int(v)
                    h = (b - a + 1) // 2
                    self.halfw.append(h)
                    self.series[(dev, d)].add(a + (b - a) // 2, v, h)
                self.samples += 1
            elif kind == "E":
                self._new_tick(int(p[1]), None)
                self.errors += 1
            elif kind == "D":
                idx, name = int(p[1]), p[2]
                while len(self.devices) <= idx:
                    self.devices.append("")
                self.devices[idx] = name
                for d in DIRS:
                    self.series.setdefault((name, d), Series(self.keep_s))
            elif kind == "H":
                self.interval_us = int(p[2])
            elif kind == "C":
                self.anchor = (int(p[1]), int(p[2]))
        except (ValueError, IndexError, KeyError):
            self.errors += 1

    def _new_tick(self, tick: int, t0: int | None) -> None:
        if tick == self.tick:
            return
        if self.tick is not None and self.track_peaks:
            self._close_tick()
        self.tick = tick
        if t0 is not None:
            self.tick_t0.append(t0)

    def _close_tick(self) -> None:
        live = [tr for tr in self.trackers.values() if not tr.frozen]
        for tr in live:                                 # take a starting sample if none yet
            for key, ser in self.series.items():
                if key not in tr.base and len(ser):
                    tr.base[key] = (ser.t[-1], ser.v[-1])
                elif key in tr.base and len(ser) and ser.v[-1] < tr.base[key][1]:
                    tr.base[key] = (ser.t[-1], ser.v[-1])      # counter was reset
        busiest = None
        for w in self.windows_ms:
            w_ns = w * 1_000_000
            for d in DIRS:
                parts = []
                for dev in self.devices:
                    r = self.series[(dev, d)].now(w_ns)
                    if r is None:
                        parts = None
                        break
                    if self.device_limit_gbps and r.lo > self.device_limit_gbps:
                        self.impossible += 1
                    for tr in live:
                        self._peak(tr, (dev, d, w), r)
                    parts.append(r)
                if parts:
                    total = sum_rates(parts)
                    for tr in live:
                        self._peak(tr, (TOTAL, d, w), total)
                    if w == self.windows_ms[0]:
                        busiest = total if busiest is None or total.gbps > busiest.gbps else busiest
        if busiest is not None:
            for tr in live:
                self._activity(tr, busiest)

    def _activity(self, tr: Tracker, r: Rate) -> None:
        start = r.t_end - r.span_ns
        if start < min((t for t, _ in tr.base.values()), default=0):
            return
        active = r.gbps > ACTIVE_GBPS
        if active and tr.first_act is None:
            tr.first_act = start
        if tr.first_act is None:
            return
        tr.hist[_bin(r.gbps, self.link_gbps)] += 1
        if active:
            tr.last_act = r.t_end
            tr.hist_last = tr.hist.copy()

    def _peak(self, tr: Tracker, key, r: Rate) -> None:
        # Only windows that START after the tracker started count, so the 1 s peak right after
        # a marker does not include the previous phase's traffic.
        # The peak is the window with the highest guaranteed minimum (lo): taking the largest
        # of thousands of central values would pick the window with the "luckiest" timing error.
        cur = tr.peaks.get(key)
        if (cur is None or r.lo > cur[0].lo) and r.t_end - r.span_ns >= min(
                (t for t, _ in tr.base.values()), default=0):
            tr.peaks[key] = (r, self.trackers["phase"].label)

    # --- trackers (live mean + peak) ----------------------------------------------

    def restart(self, name: str, label: str) -> None:
        """Start the tracker now (the latest samples become its start)."""
        tr = Tracker(label)
        for key, ser in self.series.items():
            if len(ser):
                tr.base[key] = (ser.t[-1], ser.v[-1])
        self.trackers[name] = tr

    def freeze(self, name: str) -> None:
        tr = self.trackers[name]
        if not tr.frozen:
            tr.end = {key: (ser.t[-1], ser.v[-1]) for key, ser in self.series.items() if len(ser)}

    def average(self, name: str, d: str, scope: str = TOTAL) -> dict | None:
        """Exact mean since the start (or until frozen), bytes and duration."""
        tr = self.trackers[name]
        devs = self.devices if scope == TOTAL else [scope]
        gbps, words, secs = 0.0, 0, 0.0
        for dev in devs:
            key = (dev, d)
            if key not in tr.base:
                return None
            ser = self.series[key]
            t1, v1 = tr.end[key] if tr.frozen and key in tr.end else (
                (ser.t[-1], ser.v[-1]) if len(ser) else tr.base[key])
            t0, v0 = tr.base[key]
            if t1 <= t0:
                return None
            gbps += (v1 - v0) * BITS_PER_WORD / (t1 - t0)
            words += v1 - v0
            secs = max(secs, (t1 - t0) / 1e9)
        return {"gbps": gbps, "GB": words * BYTES_PER_WORD / 1e9, "s": secs}

    def link_stats(self, name: str, target_gbps: float) -> dict | None:
        """Link bottleneck estimate for a tracker + its 5 ms rate distribution (% of the
        active span)."""
        tr = self.trackers[name]
        avgs = [self.average(name, d) for d in DIRS]
        if tr.first_act is None or tr.last_act is None or any(a is None for a in avgs):
            return None
        est = link_estimate(max(a["GB"] for a in avgs), (tr.last_act - tr.first_act) / 1e9,
                            self.link_gbps, target_gbps)
        n = sum(tr.hist_last)
        est["hist"] = [100.0 * c / n for c in tr.hist_last] if n else None
        return est

    def peak(self, name: str, d: str, w_ms: int, scope: str = TOTAL) -> tuple[Rate, str] | None:
        return self.trackers[name].peaks.get((scope, d, w_ms))

    # --- output -----------------------------------------------------------------

    def current(self, scope: str, d: str, w_ms: int) -> Rate | None:
        w_ns = w_ms * 1_000_000
        if scope != TOTAL:
            return self.series[(scope, d)].now(w_ns)
        parts = [self.series[(dev, d)].now(w_ns) for dev in self.devices]
        if not parts or any(r is None for r in parts):
            return None
        return sum_rates(parts)

    def reset_peaks(self) -> None:
        self.restart("overall", "since reset")
        self.restart("phase", "since reset")

    def last_t(self) -> int:
        return max((s.t[-1] for s in self.series.values() if len(s)), default=0)

    def wall(self, t_ns: int) -> float:
        """CLOCK_MONOTONIC_RAW ns -> epoch seconds (using the latest clock anchor)."""
        if not self.anchor:
            return 0.0
        mono, real = self.anchor
        return (real + (t_ns - mono)) / 1e9

    def health(self) -> dict:
        t0 = list(self.tick_t0)
        periods = sorted((b - a) / 1e6 for a, b in zip(t0, t0[1:]))
        hw = sorted(self.halfw)
        return {
            "tick_p50_ms": _pct(periods, 0.50),
            "tick_p95_ms": _pct(periods, 0.95),
            "halfw_p50_us": _pct(hw, 0.50) / 1e3 if hw else None,
            "halfw_p95_us": _pct(hw, 0.95) / 1e3 if hw else None,
            "errors": self.errors,
            "impossible": self.impossible,
            "resets": sum(s.resets for s in self.series.values()),
            "samples": self.samples,
        }


def _pct(xs: list, p: float):
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(p * len(xs)))]


# ----------------------------------------------------------------------------
# Recording: load from file and summarize per phase (marker)
# ----------------------------------------------------------------------------

class Recording:
    """A recording file: sampler output plus M (marker) lines."""

    def __init__(self, path: str, link_gbps: float = 200.0):
        self.path = path
        self.state = LiveState(keep_s=None, track_peaks=False, link_gbps=link_gbps)
        self.markers: list[tuple[int, str]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("M,"):
                    _, t, label = line.rstrip("\n").split(",", 2)
                    self.markers.append((int(t), label))
                else:
                    self.state.feed(line)
        self.markers.sort()

    @property
    def devices(self) -> list[str]:
        return self.state.devices

    def span(self) -> tuple[int, int]:
        ss = [s for s in self.state.series.values() if len(s)]
        if not ss:
            return 0, 0
        return max(s.t[0] for s in ss), min(s.t[-1] for s in ss)

    def segments(self) -> list[tuple[str, int, int]]:
        start, end = self.span()
        before = [lab for t, lab in self.markers if t <= start]
        cuts = [(t, lab) for t, lab in self.markers if start < t < end]
        bounds = [(start, before[-1] if before else "(start)")] + cuts + [(end, None)]
        min_ns = min(self.state.windows_ms) * 1_000_000
        return [(bounds[i][1], bounds[i][0], bounds[i + 1][0]) for i in range(len(bounds) - 1)
                if bounds[i + 1][0] - bounds[i][0] >= min_ns]


def summarize(rec: Recording, threshold_gbps: float = ACTIVE_GBPS, windows_ms=WINDOWS_MS,
              link_gbps: float = 200.0, target_gbps: float = 100.0) -> list[dict]:
    """For every phase x direction: duration, bytes, exact mean, window distributions."""
    out = []
    st = rec.state
    start, end = rec.span()
    segments = rec.segments()
    if len(segments) > 1:
        segments.append(("(whole recording)", start, end))
    for label, s, e in segments:
        for d in DIRS:
            series = [st.series[(dev, d)] for dev in st.devices]
            total_words = 0
            mean = 0.0
            for ser in series:
                i0, i1 = ser.index_at(s), ser.index_at(e)
                if i0 < 0 or i1 <= i0:
                    continue
                dv = ser.v[i1] - ser.v[i0]
                total_words += dv
                mean += dv * BITS_PER_WORD / (ser.t[i1] - ser.t[i0])
            row = {
                "phase": label,
                "direction": "TX (sent)" if d == "tx" else "RX (received)",
                "duration_s": (e - s) / 1e9,
                "GB": total_words * BYTES_PER_WORD / 1e9,
                "mean_Gbps": mean,
            }
            for w in windows_ms:
                rates = _grid(series, s, e, w * 1_000_000)
                vals = sorted(r.gbps for r in rates)
                peak = _sliding_peak(series, s, e, w * 1_000_000)
                row[f"{w}ms_p50"] = _pct(vals, 0.50)
                row[f"{w}ms_p99"] = _pct(vals, 0.99)
                row[f"{w}ms_peak_min"] = peak.lo if peak else None
                row[f"{w}ms_peak"] = peak.gbps if peak else None
                row[f"{w}ms_peak_±"] = peak.pm if peak else None
                if w == windows_ms[0]:
                    row[f"active_%({w}ms>{threshold_gbps}Gbps)"] = (
                        100.0 * sum(v > threshold_gbps for v in vals) / len(vals) if vals else None)
                    row.update(_link_columns(rates, row["GB"], threshold_gbps, link_gbps, target_gbps))
            out.append(row)
    return out


def _link_columns(rates: list[Rate], gb: float, threshold: float,
                  link_gbps: float, target_gbps: float) -> dict:
    """Link bottleneck columns for a phase (from consecutive 5 ms windows)."""
    act = [i for i, r in enumerate(rates) if r.gbps > threshold]
    cols = {"busy_s": None, "active_span_s": None, "link_busy_%": None,
            f"time_factor_{target_gbps:g}G": None}
    cols.update({f"%time_{lab}": None for lab in HIST_LABELS})
    if not act:
        return cols
    first, last = act[0], act[-1]
    span_s = (rates[last].t_end - (rates[first].t_end - rates[first].span_ns)) / 1e9
    est = link_estimate(gb, span_s, link_gbps, target_gbps)
    hist = [0] * len(HIST_LABELS)
    for r in rates[first:last + 1]:
        hist[_bin(r.gbps, link_gbps)] += 1
    n = last - first + 1
    cols.update({"busy_s": est["busy_s"], "active_span_s": span_s,
                 "link_busy_%": 100 * est["f"], f"time_factor_{target_gbps:g}G": est["factor"]})
    cols.update({f"%time_{lab}": 100.0 * c / n for lab, c in zip(HIST_LABELS, hist)})
    return cols


def _grid(series: list[Series], s: int, e: int, w_ns: int) -> list[Rate]:
    """Consecutive (non-overlapping) W windows in [s, e]; sum over all devices."""
    rates = []
    end = s + w_ns
    while end <= e:
        parts = []
        for ser in series:
            i = ser.index_at(end)
            r = ser.window(i, w_ns)
            if r is None or ser.t[i] - r.span_ns < s:
                parts = None
                break
            parts.append(r)
        if parts:
            rates.append(sum_rates(parts))
        end += w_ns
    return rates


def _sliding_peak(series: list[Series], s: int, e: int, w_ns: int) -> Rate | None:
    """Highest sliding W window that starts and ends inside [s, e] (same definition as the
    live peak). Long windows are evaluated every W/10 ms instead of at every sample."""
    if not series or not len(series[0]):
        return None
    ref = series[0]
    stride = max(1, w_ns // 10_000_000)
    best = None
    i = ref.index_at(s + w_ns)
    last = ref.index_at(e)
    while 0 <= i <= last:
        t = ref.t[i]
        parts = []
        for ser in series:
            r = ser.window(ser.index_at(t), w_ns)
            if r is None or r.t_end - r.span_ns < s:
                parts = None
                break
            parts.append(r)
        if parts:
            tot = sum_rates(parts)
            if best is None or tot.lo > best.lo:
                best = tot
        i += stride
    return best
