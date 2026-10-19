"""rdma_sampler tests, on a fake sysfs tree (no hardware needed).

    python3 -m unittest discover -s tests -v
"""

import os
import shutil
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "netinfermon", "sampler", "rdma_sampler.c")


def make_dev(root, name, state, tx, rx):
    d = os.path.join(root, name, "ports", "1")
    os.makedirs(os.path.join(d, "counters"))
    with open(os.path.join(d, "state"), "w") as f:
        f.write(state + "\n")
    for fn, v in (("port_xmit_data", tx), ("port_rcv_data", rx)):
        with open(os.path.join(d, "counters", fn), "w") as f:
            f.write(f"{v}\n")


class SamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.bin = os.path.join(cls.tmp, "rdma_sampler")
        subprocess.run(["gcc", "-O2", "-Wall", "-Wextra", "-Werror", "-o", cls.bin, SRC], check=True)
        cls.root = os.path.join(cls.tmp, "ib")
        make_dev(cls.root, "rocep1s0f1", "4: ACTIVE", 1000, 2000)
        make_dev(cls.root, "rocep1s0f0", "1: DOWN", 5, 6)
        make_dev(cls.root, "roceP2p1s0f1", "4: ACTIVE", 3000, 4000)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def run_sampler(self, *args):
        out = subprocess.run([self.bin, "-b", self.root, *args], capture_output=True, text=True,
                             timeout=10)
        return out.returncode, out.stdout.splitlines(), out.stderr

    def test_only_active_devices_sorted(self):
        rc, lines, _ = self.run_sampler("-n", "3")
        self.assertEqual(rc, 0)
        self.assertEqual([l for l in lines if l.startswith("D,")],
                         ["D,0,roceP2p1s0f1", "D,1,rocep1s0f1"])
        self.assertTrue(lines[0].startswith("H,2,1000,2"))

    def test_sample_lines_and_timestamps(self):
        rc, lines, _ = self.run_sampler("-n", "20", "-i", "1000")
        s = [l.split(",") for l in lines if l.startswith("S,")]
        self.assertEqual(len(s), 40)
        for p in s:
            ta, tb, tx, ra, rb, rx = map(int, p[3:9])
            self.assertLessEqual(ta, tb)
            self.assertLessEqual(tb, ra)
            self.assertLessEqual(ra, rb)
        self.assertEqual((s[0][5], s[0][8]), ("3000", "4000"))   # idx 0 = roceP2p1s0f1
        self.assertEqual((s[1][5], s[1][8]), ("1000", "2000"))
        starts = [int(p[3]) for p in s if p[2] == "0"]
        periods = sorted(b - a for a, b in zip(starts, starts[1:]))
        self.assertLess(abs(periods[len(periods) // 2] - 1_000_000), 300_000, periods)

    def test_failed_read_emits_error_not_sample(self):
        bad = os.path.join(self.root, "rocep1s0f1", "ports", "1", "counters", "port_xmit_data")
        with open(bad, "w") as f:
            f.write("garbage\n")
        try:
            rc, lines, _ = self.run_sampler("-n", "2")
        finally:
            with open(bad, "w") as f:
                f.write("1000\n")
        self.assertEqual(rc, 0)
        self.assertEqual([l for l in lines if l.startswith("E,")], ["E,0,1,tx,22", "E,1,1,tx,22"])
        self.assertTrue(all(l.split(",")[2] == "0" for l in lines if l.startswith("S,")))

    def test_explicit_device_list(self):
        rc, lines, _ = self.run_sampler("-n", "1", "-d", "rocep1s0f0")
        self.assertEqual([l for l in lines if l.startswith("D,")], ["D,0,rocep1s0f0"])

    def test_exits_when_reader_closes(self):
        p = subprocess.Popen([self.bin, "-b", self.root, "-i", "1000"], stdout=subprocess.PIPE)
        p.stdout.readline()
        p.stdout.close()                     # the reader went away (service closed)
        self.assertIsNotNone(p.wait(timeout=2))

    def test_survives_starting_thread_exit(self):
        # The service may start it from a short-lived thread; the sampler must survive.
        import threading
        import time
        box = {}

        def starter():
            p = subprocess.Popen([self.bin, "-b", self.root, "-i", "1000"], stdout=subprocess.PIPE)
            box["first"] = p.stdout.readline()     # the sampler has fully started
            box["p"] = p                           # ... then the thread exits

        t = threading.Thread(target=starter)
        t.start()
        t.join()
        p = box["p"]
        try:
            self.assertTrue(box["first"].startswith(b"H,2,"))
            time.sleep(0.3)
            self.assertIsNone(p.poll(), "the sampler died with the thread that started it")
        finally:
            p.kill()
            p.wait()
            p.stdout.close()

    def test_no_active_device_fails_cleanly(self):
        rc, _, err = self.run_sampler("-n", "1", "-d", "missing")
        self.assertEqual(rc, 1)
        self.assertIn("no active RDMA device", err)


if __name__ == "__main__":
    unittest.main()
