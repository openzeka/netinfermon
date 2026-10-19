"""Load generator unit tests (no server)."""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from netinfermon import loadgen  # noqa: E402

GB = 1_000_000_000


class LinkCheckTests(unittest.TestCase):
    def test_too_little_traffic_defers(self):
        self.assertIsNone(loadgen.check_link("all", {"a": 0, "b": 0}, {"a": 1_000_000, "b": 1_000_000}))

    def test_all_ok_and_fail(self):
        self.assertTrue(loadgen.check_link("all", {"a": 0, "b": 0}, {"a": 5 * GB, "b": 4 * GB}).startswith("ok:"))
        self.assertTrue(loadgen.check_link("all", {"a": 0, "b": 0}, {"a": 88 * GB, "b": 12 * GB}).startswith("ok:"))
        self.assertIn("every device", loadgen.check_link("all", {"a": 0, "b": 0}, {"a": 9 * GB, "b": GB // 20}))

    def test_one_ok_and_fail(self):
        self.assertTrue(loadgen.check_link("one", {"a": 0, "b": 0}, {"a": 5 * GB, "b": GB // 1000}).startswith("ok:"))
        self.assertIn("one device only", loadgen.check_link("one", {"a": 0, "b": 0}, {"a": 5 * GB, "b": 2 * GB}))

    def test_any_never_fails(self):
        self.assertTrue(loadgen.check_link("any", {"a": 0, "b": 0}, {"a": 9 * GB, "b": 0}).startswith("ok:"))


class PromptTests(unittest.TestCase):
    def test_pad_prompt_is_seeded_and_exact(self):
        prompts = loadgen.load_prompts()
        filler = [p.replace(loadgen.OUTPUT_HINT, "").strip() for p in prompts]
        a = loadgen.pad_prompt(prompts[0], filler, 20_000, random.Random("1:P1:0"))
        b = loadgen.pad_prompt(prompts[0], filler, 20_000, random.Random("1:P1:0"))
        c = loadgen.pad_prompt(prompts[0], filler, 20_000, random.Random("1:P1:1"))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(a), 20_000)
        self.assertTrue(a.startswith(prompts[0]))

    def test_output_hint(self):
        p = f"Explain X. {loadgen.OUTPUT_HINT}"
        self.assertIn("at least 1024 tokens", loadgen.set_output_hint(p, 1024))

    def test_every_prompt_has_the_hint(self):
        self.assertTrue(all(loadgen.OUTPUT_HINT in p for p in loadgen.load_prompts()))


class PlanTests(unittest.TestCase):
    def test_builtin_plans_load(self):
        plans = loadgen.list_plans()
        self.assertIn("linkspeed", plans)
        for name in plans:
            plan = loadgen.load_plan(name)
            self.assertTrue(plan["points"])
            self.assertEqual(len({p["id"] for p in plan["points"]}), len(plan["points"]))

    def test_build_plan(self):
        plan = loadgen.build_plan([1024, 8192], 256, [1, 4], 1, 3, 5)
        self.assertEqual([p["id"] for p in plan["points"]], ["c1-in1024", "c1-in8192", "c4-in1024", "c4-in8192"])
        self.assertEqual((plan["measured_rounds"], plan["seed"]), (3, 5))

    def test_request_summary_skips_warmup_and_errors(self):
        row = {"point": "P1", "concurrency": "1", "input_target": "1024", "output_target": "64",
               "ttft_ms": "100", "decode_tps": "50", "e2e_tps": "40", "latency_s": "2", "prompt_tokens": "1024"}
        rows = [{**row, "warmup": "1", "error": ""}, {**row, "warmup": "0", "error": ""},
                {**row, "warmup": "0", "error": "", "ttft_ms": "300"}, {**row, "warmup": "0", "error": "boom"}]
        s = loadgen.request_summary(rows)[0]
        self.assertEqual((s["requests"], s["errors"], s["ttft_ms_mean"]), (3, 1, 200.0))


if __name__ == "__main__":
    unittest.main()
