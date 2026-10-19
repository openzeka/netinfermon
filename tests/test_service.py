"""Service tests: control API with a real sampler on a fake sysfs tree, and an end-to-end
benchmark run against a stand-in LLM server (no GPU, no NIC), then report and compare.

    python3 -m unittest discover -s tests -v
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
from test_sampler import make_dev  # noqa: E402

from netinfermon import analysis, compare, config, loadgen  # noqa: E402
from netinfermon.monitor import Service  # noqa: E402

LLM_PORT = 30199


def call(port, path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=None if body is None else json.dumps(body).encode(),
                                 method="GET" if body is None else "POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(dir=os.path.expanduser("~"), prefix=".netinfermon-test-")
        root = os.path.join(cls.tmp, "ib")
        make_dev(root, "rocep1s0f1", "4: ACTIVE", 1000, 2000)
        make_dev(root, "roceP2p1s0f1", "4: ACTIVE", 3000, 4000)
        for dev in ("rocep1s0f1", "roceP2p1s0f1"):
            with open(os.path.join(root, dev, "ports", "1", "rate"), "w") as f:
                f.write("200 Gb/sec (4X HDR)\n")
        cls.port = 18599
        cls.settings = config.Settings(data_dir=os.path.join(cls.tmp, "data"), sysfs_root=root,
                                       api_port=cls.port, endpoint=f"http://127.0.0.1:{LLM_PORT}/v1",
                                       link_max_gbps=195.0, one_device_gbps=109.0)
        os.environ["XDG_CACHE_HOME"] = os.path.join(cls.tmp, "cache")
        cls.svc = Service(cls.settings)
        cls.svc.start()
        cls.llm = subprocess.Popen([sys.executable, os.path.join(HERE, "fake_llm_server.py"), str(LLM_PORT)])
        time.sleep(0.8)

    @classmethod
    def tearDownClass(cls):
        cls.svc.stop()
        cls.llm.terminate()
        cls.llm.wait()
        time.sleep(0.2)
        shutil.rmtree(cls.tmp)

    def test_status(self):
        code, st = call(self.port, "/status")
        self.assertEqual(code, 200)
        self.assertTrue(st["alive"])
        self.assertEqual(st["devices"], ["roceP2p1s0f1", "rocep1s0f1"])
        self.assertEqual(st["counters"]["rocep1s0f1"]["tx_bytes"], 4000)      # 1000 words × 4
        self.assertEqual((st["link_gbps"], st["nominal_gbps"]), (195.0, 200.0))
        self.assertFalse(st["recording"])

    def test_record_marker_stop(self):
        path = os.path.join(self.tmp, "manual", "recording-1.csv")
        code, r = call(self.port, "/record/start", {"path": path})
        self.assertEqual((code, r["path"]), (200, os.path.realpath(path)))
        self.assertEqual(call(self.port, "/record/start", {})[0], 409)       # already recording
        code, m = call(self.port, "/marker", {"label": "P1 round 1/4, warm-up"})
        time.sleep(0.3)
        code, r = call(self.port, "/record/stop", {})
        with open(path) as fh:
            lines = fh.read().splitlines()
        self.assertTrue(lines[0].startswith("H,2,"))
        self.assertIn(f"M,{m['t_ns']},P1 round 1/4 warm-up", lines)      # comma removed from label
        self.assertGreater(sum(l.startswith("S,") for l in lines), 100)
        self.assertEqual(call(self.port, "/record/start", {"path": path})[0], 409)   # never overwrites
        self.assertEqual(call(self.port, "/record/start", {"path": "/tmp/x.csv"})[0], 400)

    def wait_run(self, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            st = self.svc.run_status()
            if st["state"] not in ("starting", "waiting for idle link", "running"):
                return st
            time.sleep(0.2)
        self.fail("run did not finish")

    def test_end_to_end_run_report_compare(self):
        plan = loadgen.build_plan([256, 1024], 32, [1, 2], warmup_rounds=1, measured_rounds=2, seed=7)
        plan.update(idle_seconds=1)
        for label in ("A", "B"):
            code, st = call(self.port, "/run/start", {"campaign": "t", "label": label, "plan": plan,
                                                      "expect": "any", "notes": "test"})
            self.assertEqual(code, 200, st)
            self.assertEqual(call(self.port, "/run/start", {"campaign": "t", "label": "X", "plan": plan})[0], 409)
            st = self.wait_run()
            self.assertEqual(st["state"], "done", st)
            self.assertEqual(st["progress"]["done"], st["progress"]["total"])
        rdir = self.settings.run_dir("t", "A")
        rows = loadgen.read_requests(os.path.join(rdir, "requests.csv"))
        self.assertEqual(len(rows), (1 + 1 + 2 + 2) * 3)        # requests per round × 3 rounds
        self.assertTrue(all(r["error"] == "" and r["output_ok"] == "1" for r in rows))
        self.assertTrue(all(int(r["t0_ns"]) < int(r["t_first_ns"]) <= int(r["t_last_ns"]) <= int(r["t_end_ns"])
                            for r in rows))
        prompts = [r["prompt_tokens"] for r in rows if r["point"] == "c1-in1024"]
        self.assertTrue(all(abs(int(p) - 1024) < 40 for p in prompts), prompts)   # sized on the server
        man = json.load(open(os.path.join(rdir, "manifest.json")))
        self.assertEqual((man["link_max_gbps"], man["expect"]), (195.0, "any"))
        self.assertTrue(os.path.exists(os.path.join(rdir, "network", "recording-1.csv")))

        # rerun of a finished label: nothing to do
        call(self.port, "/run/start", {"campaign": "t", "label": "A", "plan": plan, "expect": "any"})
        self.assertEqual(self.wait_run()["state"], "done")
        self.assertEqual(len(loadgen.read_requests(os.path.join(rdir, "requests.csv"))), len(rows))

        res = analysis.report(rdir)
        self.assertEqual([p["point"] for p in res["points"]], [p["id"] for p in plan["points"]])
        self.assertFalse(res["warnings"], res["warnings"])
        self.assertIn("| c1-in256 |", open(os.path.join(rdir, "report", "report.md")).read())
        analysis.report(self.settings.run_dir("t", "B"))
        cmp = compare.compare(rdir, self.settings.run_dir("t", "B"))
        self.assertEqual(len(cmp["rows"]), 4)
        self.assertFalse(cmp["warnings"])

    def test_link_check_without_traffic_gives_no_verdict(self):
        # the fake NIC carries no traffic at all, so "one" can never be confirmed: the run must
        # finish without a verdict rather than fail; a wrong split is tested in test_loadgen
        plan = loadgen.build_plan([128], 8, [1], 0, 1, 1)
        plan.update(idle_seconds=1)
        self.svc.start_run("t", "linkcheck", plan, expect="one")
        st = self.wait_run()
        self.assertEqual(st["state"], "done", st)
        man = json.load(open(os.path.join(self.settings.run_dir("t", "linkcheck"), "manifest.json")))
        self.assertNotIn("link_check", man)
        self.assertEqual(man["link_max_gbps"], 109.0)


if __name__ == "__main__":
    unittest.main()
