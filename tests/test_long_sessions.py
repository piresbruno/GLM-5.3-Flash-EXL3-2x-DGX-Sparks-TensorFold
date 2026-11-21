"""Offline tests of tools/long_sessions.py's sampling and scoring (no server; fixtures are not inference results)."""
import importlib.util
import random
import sys
import unittest
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "tools/long_sessions.py"
sys.path.insert(0, str(p.parent))
s = importlib.util.spec_from_file_location("long_sessions", p)
assert s is not None and s.loader is not None
ls = importlib.util.module_from_spec(s)
s.loader.exec_module(ls)


def turn(phase, sid, t, prompt, cached, expected, kind="turn", **kw):
    row = {"t": 1000.0 + t * 10, "phase": phase, "sid": sid, "turn": t, "kind": kind, "busy": 0, "prompt": prompt,
           "completion": 100, "cached": cached, "prefill_s": 1.0, "decode_s": 1.0, "ttft": 1.0, "wall": 2.0,
           "expected": expected, "miss": max(0, expected - cached),
           "resume": min(1.0, cached / expected) if expected else None}
    row.update(kw)
    return row


def mon(t, phase, avail, kept=0):
    return {"t": t, "phase": phase, "mem": {"local": {"avail_gib": avail, "total_gib": 121.6, "rss_gib": 13.0}},
            "health": {"kept_bytes": kept, "streams": {"paused": 0}}}


H = {"kept_bytes_cap": 4 << 30, "spill": {}}


class SamplingTests(unittest.TestCase):
    def test_sample_stays_inside_the_quantiles(self):
        rng = random.Random(1)
        xs = [ls.sample(ls.INCREMENTS, rng) for _ in range(20000)]
        self.assertGreaterEqual(min(xs), ls.INCREMENTS[0])
        self.assertLessEqual(max(xs), ls.INCREMENTS[-1])
        med = sorted(xs)[len(xs) // 2]
        self.assertTrue(ls.INCREMENTS[4] < med < ls.INCREMENTS[6], med)   # around production's p50 (1984)

    def test_sample_cap(self):
        rng = random.Random(2)
        self.assertTrue(all(ls.sample(ls.INCREMENTS, rng, 500) <= 500 for _ in range(1000)))

    def test_expected_cached_is_the_grid_point_below(self):
        self.assertEqual(ls.expected_cached(0), 0)
        self.assertEqual(ls.expected_cached(63), 0)
        self.assertEqual(ls.expected_cached(130_000), 129_984)

    def test_phases_spec(self):
        self.assertEqual(ls.parse_phases("warm:2:5,idle:0:0.5"), [("warm", 2, 5.0), ("idle", 0, 0.5)])

    def test_slope(self):
        pts = [(i * 360.0, 20 - i * 0.2) for i in range(10)]          # -0.2 GiB every 6 min = -2 GiB/h
        self.assertAlmostEqual(ls.slope_gib_h(pts), -2.0)
        self.assertIsNone(ls.slope_gib_h(pts[:3]))
        self.assertIsNone(ls.slope_gib_h(pts, min_span=3600))         # 54 minutes of samples: too short

    def test_a_cut_off_recall_warns_but_does_not_fail(self):
        rows = [{"t": 1, "phase": "steady", "sid": 0, "kind": "recall", "correct": False, "truncated": True,
                 "depth": 30000, "prompt": 30000, "cached": 0, "busy": 0}]
        s = ls.summarize(rows, [mon(0, "steady", 15)], H, H, 10, None)
        self.assertNotIn("recall correct", s["failed"])
        self.assertTrue(any("recalls spent" in w for w in s["warnings"]))


class SummaryTests(unittest.TestCase):
    def test_clean_run_passes(self):
        rows = [turn("steady", 0, 0, 20000, 10816, 0), turn("steady", 0, 1, 30000, 19968, 19968),
                {"t": 1, "phase": "steady", "sid": 0, "kind": "recall", "correct": True, "depth": 30000,
                 "prompt": 30000, "cached": 30000, "busy": 0}]
        m = [mon(i * 10.0, "steady", 15 - i * 0.001) for i in range(10)]
        s = ls.summarize(rows, m, H, H, 100, None)
        self.assertEqual(s["failed"], [], s["checks"])
        self.assertEqual(s["phases"]["steady"]["turns"], 2)

    def test_the_157k_re_prefill_fails(self):
        rows = [turn("steady", 3, 5, 186775, 28992, 186752)]          # 2026-11-12's conversation 78
        s = ls.summarize(rows, [mon(0, "steady", 15)], H, H, 100, None)
        self.assertIn(f"no {ls.BIG_MISS}+ token re-prefills outside overload", s["failed"])
        self.assertEqual(s["big_misses"][0]["miss"], 186752 - 28992)

    def test_misses_under_overload_are_not_failures_but_recall_and_errors_are(self):
        rows = [turn("overload", 1, 2, 100000, 50000, 99968),
                {"t": 1, "phase": "overload", "sid": 1, "kind": "recall", "correct": False, "depth": 100000,
                 "prompt": 100000, "cached": 0, "busy": 0},
                {"t": 2, "phase": "overload", "sid": 2, "turn": 1, "kind": "turn", "error": "HTTP 500"}]
        s = ls.summarize(rows, [mon(0, "overload", 15)], H, H, 100, None)
        self.assertNotIn(f"no {ls.BIG_MISS}+ token re-prefills outside overload", s["failed"])
        self.assertIn("recall correct", s["failed"])
        self.assertIn("no request errors", s["failed"])

    def test_memory_decay_and_429_outside_overload(self):
        rows = [turn("steady", 0, 1, 30000, 29952, 29952, busy=3)]
        m = [mon(i * 360.0, "steady", 20 - i * 0.5) for i in range(10)]   # -5 GiB/h over 54 minutes
        s = ls.summarize(rows, m, H, H, 3600, None)
        self.assertTrue(any("MemAvailable falls" in w for w in s["warnings"]), s["warnings"])
        self.assertIn("no 429 outside overload", s["failed"])

    def test_guard_and_spill_failures(self):
        h1 = {"spill": {"crc_failed": 1}}
        s = ls.summarize([], [mon(0, "steady", 4)], H, h1, 10, [("local", 4)])
        self.assertIn("no spill write/load/crc failures", s["failed"])
        self.assertIn(f"memory stayed above {ls.GUARD_GIB} GiB", s["failed"])


if __name__ == "__main__":
    unittest.main()
