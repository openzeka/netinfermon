"""analysis tests: a synthetic recording with known traffic, requests on the same clock.

    python3 -m unittest discover -s tests -v
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
from test_rates import simulate  # noqa: E402  (traffic simulator)

from netinfermon import analysis  # noqa: E402

MS = 1_000_000
T0 = 5 * MS                      # simulate() offsets its clock by 5 ms


def ns(t_ms):
    return T0 + int(t_ms * MS)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.dir, "network"))
        # warm-up round 1000–1600 ms (prefill 1000–1100 at 150 Gb/s), measured round:
        # prefill 2000–2500 ms at 150 Gb/s with a 200 ms stall (2200–2400), generation 2500–3500 ms at 2 Gb/s
        fn = lambda t: (150.0 if (1000 <= t < 1100 or 2000 <= t < 2200 or 2400 <= t < 2500)
                        else (2.0 if 2500 <= t < 3500 else 0.0))
        lines, self.tr = simulate(fn, 4000)
        with open(os.path.join(self.dir, "network", "recording-1.csv"), "w") as f:
            f.writelines(lines)
        base = {"label": "T", "point": "P1", "slot": 0, "concurrency": 1, "error": "",
                "ttft_ms": "500.0", "decode_tps": "95.5"}
        self.rows = [
            {**base, "round": "0", "warmup": "1", "prompt_tokens": "1000",
             "t0_ns": ns(1000), "t_first_ns": ns(1100), "t_end_ns": ns(1600)},
            {**base, "round": "1", "warmup": "0", "prompt_tokens": "10000",
             "t0_ns": ns(2000), "t_first_ns": ns(2500), "t_end_ns": ns(3500)},
        ]
        self.plan = {"points": [{"id": "P1", "concurrency": 1, "input_tokens": 10000, "output_tokens": 100}]}

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_prefill_and_generation_numbers(self):
        rounds, points, ceiling = analysis.analyze(self.dir, self.rows, self.plan, 195.0)
        r = next(x for x in rounds if x["round"] == 1)
        moved = 150e9 * 0.3 / 8 / 1e9                                         # 300 ms at 150 Gb/s
        self.assertAlmostEqual(r["prefill_tx_GB"], moved, delta=0.02)
        self.assertAlmostEqual(r["prefill_mean_gbps"], 150.0 * 0.3 / 0.5, delta=0.5)
        self.assertAlmostEqual(r["prefill_bytes_per_token"], moved * 1e9 / 10000, delta=2_000)
        self.assertAlmostEqual(r["prefill_link_busy"], (moved * 8 / 195) / 0.5, delta=0.005)
        self.assertLessEqual(r["prefill_peak5_min_gbps"], 150.0)
        self.assertGreater(r["prefill_peak5_min_gbps"], 140.0)
        # per-sample: 60% of prefill at 150 Gb/s → p90 / p99 at the burst rate, p50 too
        self.assertAlmostEqual(r["prefill_1ms_p90_gbps"], 150.0, delta=15.0)
        self.assertAlmostEqual(r["prefill_1ms_p99_gbps"], 150.0, delta=25.0)
        # split: 300 ms transfer, the 200 ms quiet run between transfers is a stall, no short
        # gaps, and no quiet time at the edges (traffic starts and ends with the prefill)
        self.assertAlmostEqual(r["prefill_transfer_s"], 0.300, delta=0.011)
        self.assertAlmostEqual(r["prefill_stall_s"], 0.200, delta=0.011)
        self.assertLess(r["prefill_gap_s"], 0.011)
        self.assertLess(r["prefill_leadin_s"] + r["prefill_tail_s"], 0.011)
        self.assertAlmostEqual(r["generation_tx_GB"], 2e9 * 1.0 / 8 / 1e9, delta=0.025)   # 0.25 GB
        self.assertAlmostEqual(sum(v for k, v in r.items() if k.startswith("tx_share_")), 1.0, places=6)
        p = points["P1"]                                  # measured round only (warm-up excluded)
        self.assertAlmostEqual(p["prefill_s"], 0.5, places=6)
        self.assertLessEqual(ceiling["tx"], 150.0)
        self.assertGreater(ceiling["tx"], 140.0)

    def test_short_quiet_runs_are_gaps(self):
        # 10 ms on / 20 ms off during prefill: the quiet runs are gaps, not stalls
        fn = lambda t: 150.0 if 1000 <= t < 1600 and (t % 30) < 10 else 0.0
        lines, _ = simulate(fn, 2000)
        with open(os.path.join(self.dir, "network", "recording-1.csv"), "w") as f:
            f.writelines(lines)
        rows = [{**self.rows[1], "t0_ns": ns(1000), "t_first_ns": ns(1600), "t_end_ns": ns(1700)}]
        rounds, _, _ = analysis.analyze(self.dir, rows, self.plan, 195.0)
        r = rounds[0]
        self.assertEqual(r["prefill_stall_s"], 0.0)
        self.assertAlmostEqual(sum(r[f"prefill_{k}_s"] for k in ("leadin", "transfer", "gap", "tail")),
                               0.6, delta=0.011)
        self.assertGreater(r["prefill_gap_s"], 0.25)

    def test_quiet_edges_are_leadin_and_tail_not_stalls(self):
        # prefill 1000–1600 ms: 150 ms quiet (request handling), 10 ms bursts every 20 ms from 1150
        # (the last one ends at 1470), then 130 ms quiet before the first token. Seen on a real
        # 1K-token prefill; these are not stalls.
        fn = lambda t: 150.0 if 1150 <= t < 1480 and (t % 20) < 10 else 0.0
        lines, _ = simulate(fn, 2000)
        with open(os.path.join(self.dir, "network", "recording-1.csv"), "w") as f:
            f.writelines(lines)
        rows = [{**self.rows[1], "t0_ns": ns(1000), "t_first_ns": ns(1600), "t_end_ns": ns(1700)}]
        r = analysis.analyze(self.dir, rows, self.plan, 195.0)[0][0]
        self.assertEqual(r["prefill_stall_s"], 0.0)
        self.assertAlmostEqual(r["prefill_leadin_s"], 0.150, delta=0.011)
        self.assertAlmostEqual(r["prefill_tail_s"], 0.130, delta=0.011)
        self.assertAlmostEqual(r["prefill_transfer_s"] + r["prefill_gap_s"], 0.320, delta=0.011)

    def test_round_outside_recording_is_flagged(self):
        rows = self.rows + [{**self.rows[1], "round": "2", "t0_ns": ns(9000), "t_first_ns": ns(9100),
                             "t_end_ns": ns(9500)}]
        rounds, _, _ = analysis.analyze(self.dir, rows, self.plan, 195.0)
        self.assertEqual(next(x for x in rounds if x["round"] == 2)["note"], "not covered by a recording")

    def test_no_traffic_does_not_crash(self):
        rows = [{**self.rows[1], "t0_ns": ns(3600), "t_first_ns": ns(3700), "t_end_ns": ns(3900)}]
        rounds, points, _ = analysis.analyze(self.dir, rows, self.plan, 195.0)
        self.assertAlmostEqual(rounds[0]["prefill_tx_GB"], 0.0, places=6)
        self.assertTrue(all(v is None for k, v in points["P1"].items() if k.startswith("tx_share_")))

    def test_tick_rates_bounds_contain_truth(self):
        tk = analysis.tick_rates(os.path.join(self.dir, "network", "recording-1.csv"))
        self.assertGreater(len(tk["t"]), 3000)
        inside = [lo - 1e-3 <= 150.0 <= hi + 1e-3 for t, lo, hi in zip(tk["t"], tk["tx_lo"], tk["tx_hi"])
                  if ns(2010) < t < ns(2190)]                     # well inside a 150 Gb/s burst
        self.assertTrue(inside and all(inside), f"{inside.count(False)} of {len(inside)} outside")

    def test_html_chart_written(self):
        analysis.analyze(self.dir, self.rows, self.plan, 195.0, charts="html")
        html = os.path.join(self.dir, "report", "P1_traffic.html")
        self.assertGreater(os.path.getsize(html), 10_000)
        with open(html, encoding="utf-8") as fh:
            text = fh.read()
        # every label has text (plotly draws "new text" for an empty one; that string is also in
        # the embedded plotly.js, so the figure's annotation list is checked, not the whole file)
        i = text.index('"annotations":') + len('"annotations":')
        annotations, _ = json.JSONDecoder().raw_decode(text, i)
        self.assertTrue(annotations)
        self.assertTrue(all(a.get("text") for a in annotations), annotations)
        self.assertIn("prefill 0.50 s", text)                 # summary line has the prefill time

    @unittest.skipUnless(__import__("importlib").util.find_spec("matplotlib"), "matplotlib not installed")
    def test_svg_chart_written(self):
        analysis.analyze(self.dir, self.rows, self.plan, 195.0, charts="html+svg")
        text = open(os.path.join(self.dir, "report", "P1_traffic.svg")).read()
        self.assertIn("TTFT 0.50 s", text)                        # labels stay text in the SVG
        self.assertIn("decode 95.5 tok/s", text)


if __name__ == "__main__":
    unittest.main()
