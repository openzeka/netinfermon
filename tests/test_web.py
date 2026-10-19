"""Web page smoke test: every tab renders without an exception (Streamlit AppTest), with a
fake NIC, a stand-in LLM server and one finished run with a report."""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..")]
from test_sampler import make_dev  # noqa: E402

LLM_PORT = 30198


@unittest.skipUnless(__import__("importlib").util.find_spec("streamlit"), "streamlit not installed")
class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(dir=os.path.expanduser("~"), prefix=".netinfermon-web-")
        root = os.path.join(cls.tmp, "ib")
        make_dev(root, "rocep1s0f1", "4: ACTIVE", 1000, 2000)
        os.environ.update(NETINFERMON_SYSFS=root, NETINFERMON_DATA=os.path.join(cls.tmp, "data"),
                          NETINFERMON_API_PORT="18598", NETINFERMON_ENDPOINT=f"http://127.0.0.1:{LLM_PORT}/v1",
                          NETINFERMON_CONFIG=os.path.join(cls.tmp, "none.toml"),
                          XDG_CACHE_HOME=os.path.join(cls.tmp, "cache"))
        cls.llm = subprocess.Popen([sys.executable, os.path.join(HERE, "fake_llm_server.py"), str(LLM_PORT)])
        time.sleep(0.8)
        from netinfermon import analysis, loadgen
        from netinfermon.monitor import get_service
        svc = get_service()
        time.sleep(0.5)
        plan = loadgen.build_plan([256], 16, [1], 0, 1, 1)
        plan.update(idle_seconds=1)
        for label in ("A", "B"):
            svc.start_run("demo", label, plan)
            svc.run_thread.join(60)
            analysis.report(svc.settings.run_dir("demo", label))
        cls.svc = svc

    @classmethod
    def tearDownClass(cls):
        cls.svc.stop()
        cls.llm.terminate()
        cls.llm.wait()
        shutil.rmtree(cls.tmp)

    def test_page_renders(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(os.path.join(HERE, "..", "netinfermon", "web", "app.py"), default_timeout=60)
        at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertEqual([t.label for t in at.tabs], ["Live", "Benchmark", "Results", "Compare"])
        md = "\n".join(m.value for m in at.markdown)
        self.assertIn("demo / B", md)                  # run panel shows the last run
        self.assertIn("A → B", md.replace("demo/", "")) # compare table rendered


if __name__ == "__main__":
    unittest.main()
