"""rates.py validation tests: synthetic traffic with a known truth.

The simulator produces the lines the sampler would: every counter read takes a random time and
the counter is read at a random moment inside it (like the firmware query on real hardware).
The tests check that the +- bounds of every value cover the truth, and that the errors found in
the first version (start-up peak, read-error jumps, counter resets, short bursts split wrongly)
do not occur.

    python3 -m unittest discover -s tests -v
"""

import os
import random
import sys
import tempfile
import unittest
from itertools import accumulate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from netinfermon.rates import TOTAL, LiveState, Recording, summarize  # noqa: E402

STEP_NS = 10_000                       # simulation resolution: 10 µs
MS = 1_000_000


class Traffic:
    """True rate function for the whole port (Gb/s), split evenly over two devices."""

    def __init__(self, fn, dur_ms):
        n = int(dur_ms * MS / STEP_NS) + 2
        # bits moved per step = Gb/s × ns
        self.cum_bits = [0.0] + list(accumulate(fn(i * STEP_NS / MS) * STEP_NS for i in range(n)))

    def bits(self, t_ns: float) -> float:
        i = int(t_ns // STEP_NS)
        frac = (t_ns - i * STEP_NS) / STEP_NS
        return self.cum_bits[i] + (self.cum_bits[i + 1] - self.cum_bits[i]) * frac

    def avg(self, t0: float, t1: float) -> float:
        return (self.bits(t1) - self.bits(t0)) / (t1 - t0)

    def max_avg(self, w_ms: float, t_from: float = 0, t_to: float | None = None) -> float:
        w = w_ms * MS
        t_to = t_to if t_to is not None else (len(self.cum_bits) - 2) * STEP_NS
        best, t = 0.0, t_from
        while t + w <= t_to:
            best = max(best, self.avg(t, t + w))
            t += STEP_NS * 10
        return best


def simulate(fn, dur_ms, tick_ms=1.0, read_us=(80, 300), seed=1, fail_ticks=(),
             reset_at_ms=None, markers=()):
    """Returns lines in the sampler format + the true traffic object."""
    rnd = random.Random(seed)
    tr = Traffic(fn, dur_ms)
    t0 = 5 * MS                                   # monotonic clock offset
    lines = ["H,2,1000,2\n", "D,0,rocep1s0f1\n", "D,1,roceP2p1s0f1\n", f"C,{t0},1791800000000000000\n"]
    marks = sorted((t0 + int(m * MS), lab) for m, lab in markers)
    reset_off = 0.0
    tick, start = 0, 0.0
    while True:
        t = start
        if t + 4 * read_us[1] * 1000 > dur_ms * MS:
            break
        for idx in (0, 1):
            vals = []
            for _ in ("tx", "rx"):
                d = rnd.uniform(*read_us) * 1000
                inst = t + rnd.uniform(0, d)
                if reset_at_ms is not None and inst >= reset_at_ms * MS and not reset_off:
                    reset_off = tr.bits(inst) / 2
                words = int((tr.bits(inst) / 2 - reset_off) // 32)
                vals.append((int(t0 + t), int(t0 + t + d), words))
                t += d + 500
            if tick in fail_ticks and idx == 1:
                lines.append(f"E,{tick},{idx},tx,5\n")
            else:
                (a1, b1, v1), (a2, b2, v2) = vals
                lines.append(f"S,{tick},{idx},{a1},{b1},{v1},{a2},{b2},{v2}\n")
        while marks and marks[0][0] <= t0 + t:
            m = marks.pop(0)
            lines.append(f"M,{m[0]},{m[1]}\n")
        tick += 1
        start = max(start + tick_ms * MS, t)
    return lines, tr


def run_live(lines, on_marker=None):
    st = LiveState()
    checks = []
    for line in lines:
        if line.startswith("M,"):
            if on_marker:
                on_marker(st, line.rstrip("\n").split(",", 2)[2])
            continue
        st.feed(line)
        if line.startswith("S,") and line.split(",")[2] == "1":   # the tick's last device arrived
            for w in (5, 100, 1000):
                r = st.current(TOTAL, "tx", w)
                if r is not None:
                    checks.append((w, r))
    st._close_tick()                        # also process the last tick's peaks
    return st, checks


class EngineTests(unittest.TestCase):

    def assert_bounds_hold(self, checks, truth_fn):
        bad = [(w, r) for w, r in checks if not (r.lo - 1e-4 <= truth_fn(r) <= r.hi + 1e-4)]
        self.assertFalse(bad, f"true value outside the bounds in {len(bad)} windows: {bad[:3]}")

    def test_steady_rate_exact_and_bounded(self):
        lines, _ = simulate(lambda t: 50.0, 3000)
        st, checks = run_live(lines)
        self.assertTrue(checks)
        self.assert_bounds_hold(checks, lambda r: 50.0)
        for w in (5, 100, 1000):
            pk = st.peak("overall", "tx", w)[0]
            self.assertLessEqual(pk.lo, 50.0)
            self.assertLess(abs(pk.gbps - 50.0) / 50.0, 0.12 if w == 5 else 0.01, (w, pk))
        self.assertEqual(st.impossible, 0)

    def test_heavy_read_jitter_bounds_still_hold(self):
        # 0.1–1.5 ms reads (worse than the contended case measured with v1)
        lines, _ = simulate(lambda t: 50.0, 3000, read_us=(100, 1500), seed=7)
        st, checks = run_live(lines)
        self.assert_bounds_hold(checks, lambda r: 50.0)
        for w in (5, 100, 1000):
            self.assertLessEqual(st.peak("overall", "tx", w)[0].lo, 50.0)

    def test_no_startup_artifact(self):
        # Traffic already flows at start-up: 20 ms × 100 Gb/s every 200 ms (v1: 1 s peak = 100)
        fn = lambda t: 100.0 if (t % 200) < 20 else 0.0
        lines, tr = simulate(fn, 4000)
        st, _ = run_live(lines)
        for w in (100, 1000):
            pk = st.peak("overall", "tx", w)[0]
            true_max = tr.max_avg(w)
            self.assertLessEqual(pk.lo, true_max * 1.02, (w, pk, true_max))
            self.assertGreater(pk.gbps, true_max * 0.9, (w, pk, true_max))

    def test_single_burst_after_idle(self):
        fn = lambda t: 100.0 if 2000 <= t < 2050 else 0.0
        lines, tr = simulate(fn, 4000)
        st, _ = run_live(lines)
        pk100 = st.peak("overall", "tx", 100)[0]
        self.assertAlmostEqual(pk100.gbps, tr.max_avg(100), delta=tr.max_avg(100) * 0.03)
        pk5 = st.peak("overall", "tx", 5)[0]
        self.assertLessEqual(pk5.lo, 100.0)
        self.assertGreater(pk5.gbps, 80.0)

    def test_short_bursts_not_divided_by_gap(self):
        # decode-like 2 ms burst every 50 ms (v1's batch mode halved the peak)
        fn = lambda t: 100.0 if (t % 50) < 2 else 0.0
        lines, tr = simulate(fn, 4000)
        st, _ = run_live(lines)
        pk5 = st.peak("overall", "tx", 5)[0]
        true5 = tr.max_avg(5)                          # 2 ms × 100 / 5 ms = 40
        self.assertLessEqual(pk5.lo, true5 * 1.01, pk5)
        self.assertGreater(pk5.gbps, true5 * 5 / 6.5, pk5)   # the window can span 5–6 ms
        self.assertAlmostEqual(st.peak("overall", "tx", 1000)[0].gbps, 4.0, delta=0.2)

    def test_failed_reads_do_not_spike(self):
        fails = set(range(500, 3000, 97))
        lines, _ = simulate(lambda t: 50.0, 3500, fail_ticks=fails)
        st, checks = run_live(lines)
        self.assertEqual(st.errors, sum(l.startswith("E,") for l in lines))
        self.assertGreater(st.errors, 20)
        self.assert_bounds_hold(checks, lambda r: 50.0)
        for w in (5, 100, 1000):
            self.assertLessEqual(st.peak("overall", "tx", w)[0].lo, 50.0)

    def test_counter_reset_does_not_spike(self):
        lines, _ = simulate(lambda t: 50.0, 3000, reset_at_ms=1500)
        st, checks = run_live(lines)
        self.assertGreaterEqual(st.health()["resets"], 2)
        self.assert_bounds_hold(checks, lambda r: 50.0)
        self.assertLessEqual(st.peak("overall", "tx", 5)[0].lo, 50.0)

    def test_recording_summary_exact(self):
        fn = lambda t: 30.0 if t < 1500 else (90.0 if (t % 100) < 40 else 0.0)
        lines, tr = simulate(fn, 4000, markers=[(100, "A steady"), (1500, "B bursty"), (3900, "end")])
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.writelines(lines)
        try:
            rows = summarize(Recording(f.name))
        finally:
            os.unlink(f.name)
        by = {r["phase"]: r for r in rows if r["direction"].startswith("TX")}
        self.assertEqual(list(by), ["(start)", "A steady", "B bursty", "end", "(whole recording)"])
        self.assertAlmostEqual(by["(whole recording)"]["GB"], sum(by[k]["GB"] for k in list(by)[:-1]), delta=0.01)
        a, b = by["A steady"], by["B bursty"]
        self.assertAlmostEqual(a["mean_Gbps"], 30.0, delta=0.3)
        self.assertAlmostEqual(b["mean_Gbps"], tr.avg(1500 * MS, 3900 * MS), delta=0.5)
        self.assertAlmostEqual(b["GB"], (tr.bits(3900 * MS) - tr.bits(1500 * MS)) / 8e9, delta=0.02)
        self.assertAlmostEqual(b["100ms_peak"], 36.0, delta=1.5)        # 40 ms × 90 / 100 ms
        self.assertLessEqual(b["5ms_peak_min"], 90.0 * 1.001)         # the lower bound never exceeds the line rate
        self.assertAlmostEqual(b["active_%(5ms>0.5Gbps)"], 40.0, delta=6.0)

    def test_tiny_segments_dropped(self):
        lines, _ = simulate(lambda t: 10.0, 1000, markers=[(1, "hemen"), (500, "orta")])
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.writelines(lines)
        try:
            labels = [lab for lab, _, _ in Recording(f.name).segments()]
        finally:
            os.unlink(f.name)
        self.assertEqual(labels, ["hemen", "orta"])


    def test_running_average_exact(self):
        fn = lambda t: 80.0 if (t % 100) < 25 else 0.0          # ortalama 20 Gb/s
        lines, tr = simulate(fn, 3000)
        st, _ = run_live(lines)
        avg = st.average("overall", "tx")
        base_t = min(t for t, _ in st.trackers["overall"].base.values()) - 5 * MS
        end_t = st.last_t() - 5 * MS
        self.assertAlmostEqual(avg["gbps"], tr.avg(base_t, end_t), delta=0.15)
        self.assertAlmostEqual(avg["GB"], (tr.bits(end_t) - tr.bits(base_t)) / 8e9, delta=0.01)

    def test_phase_trackers_and_freeze(self):
        # 0–1.5 s: steady 60 Gb/s; marker; 1.5–3 s: steady 10 Gb/s; recording stops at 3 s
        fn = lambda t: 60.0 if t < 1500 else 10.0
        lines, _ = simulate(fn, 4000, markers=[(1500, "low"), (3000, "stop")])

        def on_marker(st, label):
            if label == "stop":
                st.freeze("overall")
                st.freeze("phase")
            else:
                st.restart("phase", label)

        st, _ = run_live(lines, on_marker)
        seg = st.average("phase", "tx")
        self.assertEqual(st.trackers["phase"].label, "low")
        self.assertAlmostEqual(seg["gbps"], 10.0, delta=0.1)          # the previous phase does not leak in
        self.assertAlmostEqual(seg["s"], 1.5, delta=0.02)               # until it was frozen
        for w in (5, 100, 1000):
            pk, where = st.peak("phase", "tx", w)
            self.assertLess(pk.lo, 10.0 * 1.01, (w, pk))                # the 60 Gb/s phase does not leak in
            self.assertEqual(where, "low")
        gpk, gwhere = st.peak("overall", "tx", 100)
        self.assertAlmostEqual(gpk.gbps, 60.0, delta=1.0)
        self.assertEqual(gwhere, "since start")                     # the peak was in the first phase
        g = st.average("overall", "tx")
        self.assertAlmostEqual(g["gbps"], (60 * 1.5 + 10 * 1.5) / 3.0, delta=0.4)


    def test_peak_lower_bound_never_exceeds_truth_under_noise(self):
        # Steady 50 Gb/s, very noisy reads, long recording: the central peak may drift up,
        # but the lower bound (the reported value) must never exceed the truth.
        lines, _ = simulate(lambda t: 50.0, 6000, read_us=(50, 1200), seed=11)
        st, _ = run_live(lines)
        for w in (5, 100, 1000):
            pk = st.peak("overall", "tx", w)[0]
            self.assertLessEqual(pk.lo, 50.0)
            self.assertGreater(pk.lo, 50.0 * (0.80 if w == 5 else 0.99))

    def test_segment_label_when_first_sample_follows_marker(self):
        lines, _ = simulate(lambda t: 5.0, 800)
        first_s = next(l for l in lines if l.startswith("S,"))
        t_first = int(first_s.split(",")[3])
        lines.insert(lines.index(first_s), f"M,{t_first - 1000},recording started\n")
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.writelines(lines)
        try:
            labels = [lab for lab, _, _ in Recording(f.name).segments()]
        finally:
            os.unlink(f.name)
        self.assertEqual(labels, ["recording started"])


    def test_link_busy_estimate_on_off_traffic(self):
        # prefill-like: 30 ms at full speed (195 Gb/s) every 100 ms from 1.0 to 2.93 s, then idle.
        # Expected: busy 0.6 s, active span ~1.93 s, f ~31%, factor at 100G = 1 + f × (195/100 − 1)
        fn = lambda t: 195.0 if 1000 <= t < 2930 and (t % 100) < 30 else 0.0
        lines, _ = simulate(fn, 3500, markers=[(900, "prefill")])
        st, _ = run_live(lines, lambda st, lab: st.restart("phase", lab))
        st.link_gbps = 195.0
        est = st.link_stats("phase", target_gbps=100.0)
        self.assertAlmostEqual(est["busy_s"], 0.6, delta=0.01)
        self.assertAlmostEqual(est["active_s"], 1.93, delta=0.02)
        self.assertAlmostEqual(est["f"], 0.6 / 1.93, delta=0.01)
        self.assertAlmostEqual(est["factor"], 1 + 0.95 * 0.6 / 1.93, delta=0.01)
        hist = dict(zip(("<1", "<25%", "25–50%", "50–75%", "≥75%"), est["hist"]))
        self.assertGreater(hist["≥75%"], 20.0)          # the full-speed bin is filled ...
        self.assertGreater(hist["<1"], 55.0)            # ... and the gaps: an on/off pattern
        self.assertLess(hist["25–50%"] + hist["50–75%"], 12.0)

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.writelines(lines)
        try:
            rows = summarize(Recording(f.name), link_gbps=195.0, target_gbps=100.0)
        finally:
            os.unlink(f.name)
        row = next(r for r in rows if r["phase"] == "prefill" and r["direction"].startswith("TX"))
        self.assertAlmostEqual(row["busy_s"], 0.6, delta=0.01)
        self.assertAlmostEqual(row["active_span_s"], 1.93, delta=0.02)
        self.assertAlmostEqual(row["link_busy_%"], 100 * 0.6 / 1.93, delta=1.0)
        self.assertAlmostEqual(row["time_factor_100G"], 1 + 0.95 * 0.6 / 1.93, delta=0.01)
        self.assertGreater(row["%time_≥75%"], 20.0)

    def test_link_histogram_flags_non_line_rate_traffic(self):
        # Steady 60 Gb/s: f is still computed (~31%) but the distribution sits in the 25–50% bin;
        # the traffic is not at full speed, so the factor is only an upper bound.
        lines, _ = simulate(lambda t: 60.0, 2000)
        st, _ = run_live(lines)
        st.link_gbps = 195.0
        est = st.link_stats("overall", target_gbps=100.0)
        self.assertAlmostEqual(est["f"], 60 / 195, delta=0.01)
        self.assertGreater(est["hist"][2], 95.0)        # "25–50%" of 195


if __name__ == "__main__":
    unittest.main()
