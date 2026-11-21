#!/usr/bin/env python3
"""Long agent sessions under load: context growth, kept-state reuse, memory and queue pressure over a timed run.

Usage: tools/long_sessions.py [--phases SPEC] [--grow-to N] [--time-scale F] [--replay DIR] [--out DIR]
       (API_URL / PORT as in client.py; PORT=8081 for scripts/local.sh's server)

Shaped on production's own sessions: the 149 prompt states rank 0's spill store held on 2026-11-19 (30 conversations,
up to 230k tokens, lineages up to 82 turns deep) and the ~/.omp agent sessions that ran against this server up to
2026-11-15 (300k-680k-token sessions; the incidents of 11-12 and 11-15). What those showed, and what this checks:

  * later turns of 160k-190k-token conversations re-prefilled 157k-180k tokens (90-110 s waits): every later turn's
    cached tokens against the prefix it shares with the turn before (the ``resume`` share, and ``big misses``);
  * unified memory ran down until earlyoom SIGTERM'd rank 0: MemAvailable on every Spark through the run, its drift,
    and a guard that stops sending (exit 3) at GUARD_GIB, well above earlyoom's -m 2 line, so the test never kills it;
  * kept_bytes growth toward its cap, spill near its cap, write/load/crc failures, paused streams, 429s past
    PARALLEL + MAX_QUEUED;
  * long-context recall: each synthetic session hides a fact in its first tool result and asks for it on a side
    branch (a fork that is not appended) every RECALL_EVERY turns, the deeper the session the deeper the fact.

Phases (--phases, default warm:2:5,steady:8:30,overload:12:10,idle:0:5): name:sessions:minutes. A phase keeps that many
sessions taking turns; sessions above the count park with their conversation (they resume in a later phase). After
the last phase every session takes one ``return`` turn, which shows what survived the idle time (kept or spilled).
A synthetic session that passes --grow-to tokens compacts like an agent: system prompt, an 8k summary and a new fact.

--replay DIR: instead of synthetic sessions, the conversations of DIR/manifest.json (real_replay.py's layout: the
stored states' token ids in DIR/points), sent as ids to /v1/completions in their stored order, MIN_TOKENS (30000) and
MIN_TURNS (2) as there. Those ids are the owner's private prompts: nothing of them but counts is written to --out.

Output in --out (default ./long-sessions-<time>): turns.jsonl (one line a request), monitor.jsonl (health and memory
samples), summary.json; a report on stdout. Exit code 1 when a check fails, 2 when the server is not there, 3 when the
memory guard stopped the run. Run it on a server nothing else uses: it reads the server-wide /health counters.
"""
import argparse
import array
import bisect
import collections
import json
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

sys.dont_write_bytecode = True           # no tools/__pycache__ from importing client
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import API_URL, prose  # noqa: E402

MODEL = os.environ.get("MODEL", "GLM-5.3-Flash-EXL3")
GUARD_GIB = float(os.environ.get("GUARD_GIB", "5"))          # stop sending below this MemAvailable on any node
RECALL_EVERY = int(os.environ.get("RECALL_EVERY", "4"))
RECALL_TOKENS = 1024                                         # room for the reasoning before the answer (64 was not)
SLOPE_MIN_S = 1200                                           # memory drift is only judged on 20+ minutes of load
BIG_MISS = int(os.environ.get("BIG_MISS", "16384"))          # a later turn re-prefilling this many shared tokens
MIN_RESUME = float(os.environ.get("MIN_RESUME", "0.9"))      # the share of the shared prefix a later turn must reuse
GRID = 64                                                    # TF_GLM_PROMPT_GRID: kept states sit on 64-token points
MON_S = 10

# Production's 2026-11-19 store (aggregate counts only): tokens each kept turn added (1032 lineage steps), the first kept
# point (system prompt + opening), and seconds between a conversation's stored turns. Quantiles at 0, 10, ..., 90, 95, 100%.
Q = (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1)
INCREMENTS = (64, 320, 576, 896, 1408, 1984, 3008, 3904, 6208, 13632, 58496, 220416)
GAPS_S = (0, 33, 87, 141, 216, 361, 470, 750, 1101, 2814, 5215, 12153)
SYSTEM_TOKENS = 10816                                        # the agent's system block (the 60th-70th percentile point)


def sample(quantiles, rng: random.Random, cap: float = float("inf")) -> float:
    """A draw from the piecewise-linear distribution through ``quantiles`` (at Q), at most ``cap``."""
    u = rng.random()
    i = min(bisect.bisect_right(Q, u), len(Q) - 1)
    lo, hi = Q[i - 1], Q[i]
    v = quantiles[i - 1] + (quantiles[i] - quantiles[i - 1]) * (u - lo) / (hi - lo)
    return min(v, cap)


def expected_cached(previous_prompt_tokens: int) -> int:
    """What a turn appending to the previous prompt should resume: that prompt, down to its last grid point."""
    return previous_prompt_tokens // GRID * GRID


def text_of(tokens: int, seed: int) -> str:
    return prose(int(tokens / 0.82), seed)                   # client.prose comes out at ~0.82 x its length


# ---------------------------------------------------------------------------------------------------------- requests
class Busy(Exception):
    """429: more than PARALLEL + MAX_QUEUED in flight."""


def stream(path: str, body: dict, timeout: float = 7200) -> dict:
    """One streamed request: its usage, the server's timings, time to the first token and the reply text."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(API_URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise Busy() from None
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:200]}") from None
    first, text, last = None, [], {}
    with resp:
        for raw in resp:                                     # read as it arrives: ttft is the first token's time
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if "error" in chunk:
                raise RuntimeError(f"stream error: {str(chunk['error'])[:200]}")
            for c in chunk.get("choices") or []:
                delta = c.get("delta") or {}
                piece = delta.get("content") or c.get("text") or ""
                if first is None and (piece or delta.get("reasoning_content") or delta.get("reasoning")):
                    first = time.time() - t0
                text.append(piece)
            if chunk.get("usage"):
                last = chunk
    usage = last.get("usage") or {}
    tf = last.get("tensorfold") or {}
    return {"prompt": usage.get("prompt_tokens") or 0, "completion": usage.get("completion_tokens") or 0,
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
            "prefill_s": tf.get("prefill_s"), "decode_s": tf.get("decode_s"), "ttft": first,
            "wall": time.time() - t0, "text": "".join(text)}


def health() -> dict:
    return json.load(urllib.request.urlopen(API_URL + "/health", timeout=30))


# ------------------------------------------------------------------------------------------------------------ monitor
def nodes_from_local_sh() -> list[str]:
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "local.sh")
    try:
        found = re.findall(r"^WORKER\d*=\"?([^\"\s#]+)", open(here).read(), re.M)
    except OSError:
        found = []
    return ["local"] + found


MEM_LOOP = ("while :; do awk '/MemTotal|MemAvailable/{printf \"%s \", $2}' /proc/meminfo; "
            "ps -C tensorfold -o rss= | awk '{s+=$1} END{print s+0}'; sleep INTERVAL; done")


class Monitor(threading.Thread):
    """Samples /health and each node's MemAvailable and tensorfold RSS every MON_S seconds into monitor.jsonl."""

    def __init__(self, out, nodes, stop: threading.Event):
        super().__init__(daemon=True)
        self.out, self.stop, self.mem, self.procs = out, stop, {}, []
        self.phase, self.guard_tripped, self.down_since = "start", None, None
        loop = MEM_LOOP.replace("INTERVAL", str(MON_S))
        for n in nodes:
            cmd = ["bash", "-c", loop] if n == "local" else \
                ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15", n, loop]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            self.procs.append(p)
            threading.Thread(target=self._read, args=(n, p), daemon=True).start()

    def _read(self, node, p):
        for line in p.stdout:
            try:
                total, avail, rss = (int(x) for x in line.split())
            except ValueError:
                continue
            self.mem[node] = {"avail_gib": round(avail / 2**20, 2), "total_gib": round(total / 2**20, 2),
                              "rss_gib": round(rss / 2**20, 2), "t": time.time()}

    def run(self):
        while not self.stop.is_set():
            row = {"t": time.time(), "phase": self.phase, "mem": dict(self.mem)}
            try:
                h = health()
                self.down_since = None
                row["health"] = {k: h.get(k) for k in ("ok", "requests_running", "requests_total", "pool_free_tokens",
                                                      "pool_tokens", "kept_prompts", "kept_bytes", "kept_bytes_cap",
                                                      "chats_continued", "prefix_waits", "cached_tokens_total")}
                row["health"]["streams"] = h.get("streams")
                row["health"]["spill"] = {k: (h.get("spill") or {}).get(k) for k in (
                    "stored_bytes", "cap_bytes", "entries", "loaded", "loaded_tokens", "load_failed", "crc_failed",
                    "write_failed", "no_space", "dropped", "unwritten_evicted", "quota_trims", "disk_hits",
                    "disk_lookups", "restore_ms_p95", "disk_free_bytes")}
            except Exception as exc:                      # noqa: BLE001 - a down server is a finding, not a crash
                row["health_error"] = str(exc)[:200]
                self.down_since = self.down_since or time.time()
            low = [(n, m["avail_gib"]) for n, m in self.mem.items() if m["avail_gib"] < GUARD_GIB]
            if low and not self.guard_tripped:
                self.guard_tripped = low
                print(f"  !! memory guard: {low} below {GUARD_GIB} GiB available: no more requests", flush=True)
                self.stop.set()
            self.out.write(json.dumps(row) + "\n")
            self.out.flush()
            self.stop.wait(MON_S)
        self.close()

    def close(self):
        for p in self.procs:
            if p.poll() is None:
                p.terminate()


# ----------------------------------------------------------------------------------------------------------- sessions
class Session:
    """One conversation that takes turns; ``turn()`` sends one request and records it."""

    def __init__(self, sid: int, rng: random.Random):
        self.sid, self.rng, self.turns, self.prev_prompt = sid, rng, 0, 0
        self.done = False

    def gap(self, scale: float) -> float:
        return sample(GAPS_S, self.rng) * scale


class Synthetic(Session):
    PROJECTS = 2                             # sessions share one of two system prompts, as production's agents did

    def __init__(self, sid, rng, grow_to):
        super().__init__(sid, rng)
        self.grow_to, self.compactions = grow_to, 0
        self.system = text_of(SYSTEM_TOKENS, 1000 + sid % self.PROJECTS)
        self._fresh(opening=text_of(int(sample(INCREMENTS, rng, 8000)) + 200, 5000 + sid))

    def _fresh(self, opening: str):
        self.secret = f"{self.rng.choice(('amber', 'cobalt', 'violet', 'umber', 'jade'))}-{self.rng.randint(1000, 9999)}"
        # a neutral fact: a "deploy token" was refused as a credential (24 of 106 recalls on 2026-11-21), not forgotten
        fact = f"Note for later: the codename of project {self.sid} is {self.secret}. "
        self.messages = [{"role": "system", "content": self.system},
                         {"role": "user", "content": fact + opening + "\nWork through the material above."}]
        self.prev_prompt = 0

    def request(self) -> tuple[str, dict]:
        return "/v1/chat/completions", {"model": MODEL, "messages": self.messages, "reasoning_effort": "low",
                                         "max_tokens": self.rng.choice((64, 128, 256, 512))}

    def after(self, r: dict):
        self.messages.append({"role": "assistant", "content": r["text"] or "ok"})
        add = int(sample(INCREMENTS, self.rng, max(1024, self.grow_to - r["prompt"])))
        self.messages.append({"role": "user", "content": "Tool output:\n" + text_of(add, self.rng.randrange(1 << 30))})
        if r["prompt"] + add > self.grow_to:                 # an agent's compaction: a new conversation, a summary
            self.compactions += 1
            self._fresh(opening="Summary of the work so far: " + text_of(8000, self.rng.randrange(1 << 30)))

    def recall(self) -> tuple[str, dict]:
        q = {"role": "user", "content": f"What is the codename of project {self.sid}, from the note at the start? "
                                        "Reply with the codename only."}
        return "/v1/chat/completions", {"model": MODEL, "messages": self.messages + [q], "temperature": 0,
                                         "max_tokens": RECALL_TOKENS, "reasoning_effort": "low"}


class Replay(Session):
    """A stored production conversation: its states' ids in stored order, one request a state."""

    def __init__(self, sid, rng, root, states):
        super().__init__(sid, rng)
        self.root, self.states = root, states
        self.gaps = [b["mtime"] - a["mtime"] for a, b in zip(states, states[1:])] + [0]
        self.prev_ids: list[int] = []

    def ids(self, r) -> list[int]:
        a = array.array("q")
        with open(os.path.join(self.root, "points", r["key"] + ".ids"), "rb") as fh:
            a.frombytes(fh.read())
        return a[:r["n"]].tolist()

    def request(self):
        self.cur = self.ids(self.states[self.turns])
        common = 0
        for x, y in zip(self.prev_ids, self.cur):           # the prefix this state shares with the one before
            if x != y:
                break
            common += 1
        self.shared = common
        return "/v1/completions", {"model": MODEL, "prompt": self.cur, "temperature": 0,
                                   "max_tokens": int(os.environ.get("REPLAY_MAX_TOKENS", "32"))}

    def after(self, r):
        self.prev_ids = self.cur
        if self.turns >= len(self.states):
            self.done = True

    def gap(self, scale):
        return self.gaps[self.turns - 1] * scale


def replay_sessions(root: str, rng) -> list[Replay]:
    rows = json.load(open(os.path.join(root, "manifest.json")))
    conv = collections.defaultdict(list)
    for r in rows:
        conv[r["lineage"].split("|")[0]].append(r)
    picked = [sorted(v, key=lambda r: r["mtime"]) for v in conv.values()
              if len(v) >= int(os.environ.get("MIN_TURNS", "2"))
              and max(r["n"] for r in v) >= int(os.environ.get("MIN_TOKENS", "30000"))]
    picked.sort(key=lambda v: v[0]["mtime"])
    return [Replay(i, random.Random(rng.random()), root, v) for i, v in enumerate(picked)]


# ------------------------------------------------------------------------------------------------------------- runner
class Run:
    def __init__(self, args):
        self.args, self.stop, self.lock = args, threading.Event(), threading.Lock()
        os.makedirs(args.out, exist_ok=True)
        self.turns_f = open(os.path.join(args.out, "turns.jsonl"), "w")
        rng = random.Random(args.seed)
        self.pool = replay_sessions(args.replay, rng) if args.replay else []
        self.make = (lambda i: Synthetic(i, random.Random(rng.random()), args.grow_to))
        self.active: dict[int, threading.Thread] = {}
        self.current: set[int] = set()                       # the sessions this phase keeps taking turns
        self.phase, self.records = "start", []

    def record(self, row: dict):
        row.pop("text", None)                                # replies stay out of the files (replay: private)
        with self.lock:
            self.records.append(row)
            self.turns_f.write(json.dumps(row) + "\n")
            self.turns_f.flush()

    def send(self, s: Session, path: str, body: dict, kind: str) -> dict | None:
        busy = 0
        while not self.stop.is_set():
            try:
                r = stream(path, body)
                break
            except Busy:
                busy += 1
                self.stop.wait(2 + s.rng.random() * 3)
            except Exception as exc:                          # noqa: BLE001 - recorded, and the session goes on
                self.record({"t": time.time(), "phase": self.phase, "sid": s.sid, "turn": s.turns, "kind": kind,
                             "error": str(exc)[:300], "busy": busy})
                return None
        else:
            return None
        row = {"t": time.time(), "phase": self.phase, "sid": s.sid, "turn": s.turns, "kind": kind, "busy": busy, **r}
        if kind != "recall":
            shared = s.shared if isinstance(s, Replay) else (s.prev_prompt if s.turns else 0)
            exp = expected_cached(shared) if s.turns else 0
            row.update(expected=exp, miss=max(0, exp - r["cached"]),
                       resume=(min(1.0, r["cached"] / exp) if exp else None))
        return row

    def session_loop(self, s: Session):
        while not self.stop.is_set() and not s.done:
            with self.lock:
                if s.sid not in self.current and self.phase != "return":
                    break                                     # parked: above this phase's count
            path, body = s.request()
            row = self.send(s, path, body, "turn" if self.phase != "return" else "return")
            if row is None:
                if self.stop.is_set() or self.phase == "return":
                    break
                self.stop.wait(5)
                continue
            s.prev_prompt = row["prompt"]
            s.turns += 1
            s.after(row)
            self.record(row)
            print(f"  {self.phase:8s} s{s.sid:02d} t{s.turns:3d} {row['prompt']:7d} tok cached {row['cached']:7d}"
                  f" miss {row.get('miss', 0):6d} ttft {row['ttft'] or 0:6.1f}s wall {row['wall']:6.1f}s"
                  + (f" 429x{row['busy']}" if row["busy"] else ""), flush=True)
            if isinstance(s, Synthetic) and s.turns % RECALL_EVERY == 0 and not self.stop.is_set():
                path, body = s.recall()
                rr = self.send(s, path, body, "recall")
                if rr:
                    rr["correct"] = s.secret in rr["text"]
                    rr["truncated"] = not rr["correct"] and rr["completion"] >= RECALL_TOKENS
                    rr["depth"] = rr["prompt"]
                    rr["answer"] = rr["text"].strip()[-120:]  # synthetic text only (replay never recalls)
                    self.record(rr)
                    verdict = "ok" if rr["correct"] else ("TRUNCATED" if rr["truncated"] else "WRONG")
                    print(f"  {self.phase:8s} s{s.sid:02d} recall at {rr['prompt']} tok: {verdict}", flush=True)
            if self.phase == "return":
                break
            self.stop.wait(min(s.gap(self.args.time_scale), self.args.max_gap))

    def sessions_up_to(self, n: int) -> list[Session]:
        if self.args.replay:                                  # the next stored conversations not yet replayed
            return [s for s in self.pool if not s.done][:n]
        while len(self.pool) < n:
            self.pool.append(self.make(len(self.pool)))
        return self.pool[:n]

    def start_sessions(self, n: int):
        picked = self.sessions_up_to(n)
        with self.lock:
            self.current = {s.sid for s in picked}
        for s in picked:
            t = self.active.get(s.sid)
            if t is None or not t.is_alive():
                t = threading.Thread(target=self.session_loop, args=(s,), daemon=True)
                self.active[s.sid] = t
                t.start()

    def set_phase(self, name: str):
        with self.lock:
            self.phase = name
        self.mon.phase = name

    def phases(self) -> float:
        """The timed phases, then one return turn a session; the run's start time."""
        self.mon.start()
        self.stop.wait(1)
        t_run = time.time()
        for name, count, minutes in self.args.phases:
            if self.stop.is_set():
                break
            self.set_phase(name)
            print(f"== phase {name}: {count} sessions for {minutes} min", flush=True)
            end = time.time() + minutes * 60
            while time.time() < end and not self.stop.is_set():
                self.start_sessions(count)
                if self.mon.down_since and time.time() - self.mon.down_since > 60:
                    print("  !! /health has not answered for 60 s: stopping", flush=True)
                    self.stop.set()
                self.stop.wait(5)
        with self.lock:
            self.current = set()                              # every session parks after the turn it is in
        for t in list(self.active.values()):
            t.join()
        if self.stop.is_set():
            return t_run
        self.set_phase("return")
        back = [s for s in self.pool if s.turns and not s.done]
        print(f"== phase return: {len(back)} sessions, one turn each, one at a time", flush=True)
        for s in back:
            if self.stop.is_set():
                break
            self.session_loop(s)
        return t_run

    def go(self) -> int:
        try:
            h = health()
        except Exception as exc:                              # noqa: BLE001
            print(f"cannot reach {API_URL}/health ({exc}): is the server running? (./start.sh)")
            return 2
        if h.get("requests_running"):
            print(f"note: the server runs {h['requests_running']} other requests: they share the load and counters")
        self.start_health = h
        self.mon = Monitor(open(os.path.join(self.args.out, "monitor.jsonl"), "w"), self.args.nodes, self.stop)
        try:
            t_run = self.phases()
        finally:
            guard = self.mon.guard_tripped
            self.stop.set()
            self.mon.close()
        time.sleep(1)
        try:
            end_health = health()
        except Exception:                                     # noqa: BLE001
            end_health = {}
        summary = summarize(self.records, read_jsonl(os.path.join(self.args.out, "monitor.jsonl")),
                            self.start_health, end_health, time.time() - t_run, guard)
        json.dump(summary, open(os.path.join(self.args.out, "summary.json"), "w"), indent=1)
        print_report(summary, self.args.out)
        return 3 if guard else (1 if summary["failed"] else 0)


# ------------------------------------------------------------------------------------------------------------ summary
def read_jsonl(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[min(len(xs) - 1, int(len(xs) * p))], 3) if xs else None


def slope_gib_h(points: list[tuple[float, float]], min_span: float = 0) -> float | None:
    """Least-squares slope of (time s, GiB) in GiB an hour; None on fewer than 6 points or under ``min_span`` s."""
    if len(points) < 6 or points[-1][0] - points[0][0] < min_span:
        return None
    ts, vs = zip(*points)
    mt, mv = statistics.fmean(ts), statistics.fmean(vs)
    den = sum((t - mt) ** 2 for t in ts)
    return round(sum((t - mt) * (v - mv) for t, v in points) / den * 3600, 2) if den else None


def summarize(rows, mon, h0, h1, secs, guard) -> dict:
    turns = [r for r in rows if r.get("kind") in ("turn", "return") and "error" not in r]
    later = [r for r in turns if r.get("expected")]
    recalls = [r for r in rows if r.get("kind") == "recall" and "error" not in r]
    errors = [r for r in rows if "error" in r]
    phases = collections.OrderedDict()
    for r in turns:
        p = phases.setdefault(r["phase"], {"turns": 0, "prompt": 0, "prefilled": 0, "completion": 0, "busy": 0,
                                           "ttft": [], "wall": [], "decode": [], "prefill_rate": [], "t": []})
        p["turns"] += 1
        p["prompt"] += r["prompt"]
        p["prefilled"] += r["prompt"] - r["cached"]
        p["completion"] += r["completion"]
        p["busy"] += r["busy"]
        p["ttft"].append(r["ttft"])
        p["wall"].append(r["wall"])
        p["t"].append(r["t"])
        if r.get("decode_s") and r["completion"] > 1:
            p["decode"].append(r["completion"] / r["decode_s"])
        if r.get("prefill_s") and r["prompt"] - r["cached"] > 2048:
            p["prefill_rate"].append((r["prompt"] - r["cached"]) / r["prefill_s"])
    per_phase = {}
    for name, p in phases.items():
        span = max(p["t"]) - min(p["t"]) if len(p["t"]) > 1 else None
        per_phase[name] = {
            "turns": p["turns"], "prompt_tokens": p["prompt"], "prefilled_tokens": p["prefilled"],
            "reuse": round(1 - p["prefilled"] / p["prompt"], 3) if p["prompt"] else None, "http_429": p["busy"],
            "ttft_p50": pct(p["ttft"], .5), "ttft_p95": pct(p["ttft"], .95), "wall_p95": pct(p["wall"], .95),
            "decode_tok_s_p50": pct(p["decode"], .5), "prefill_tok_s_p50": pct(p["prefill_rate"], .5),
            "completion_tok_s": round(p["completion"] / span, 1) if span else None}
    nodes = sorted({n for m in mon for n in m.get("mem", {})})
    mem = {}
    for n in nodes:
        seen = [m for m in mon if n in m.get("mem", {})]
        pts = [(m["t"], m["mem"][n]["avail_gib"]) for m in seen]
        rss = [(m["t"], m["mem"][n]["rss_gib"]) for m in seen]
        busy = [(m["t"], m["mem"][n]["avail_gib"]) for m in seen if m["phase"] not in ("start", "idle", "return")]
        mem[n] = {"avail_start": pts[0][1] if pts else None, "avail_min": min(v for _, v in pts) if pts else None,
                  "avail_end": pts[-1][1] if pts else None, "avail_slope_gib_h": slope_gib_h(busy, SLOPE_MIN_S),
                  "rss_max": max(v for _, v in rss) if rss else None,
                  "rss_slope_gib_h": slope_gib_h(rss, SLOPE_MIN_S)}
    hs = [m["health"] for m in mon if "health" in m]
    kept_max = max((h["kept_bytes"] or 0 for h in hs), default=0)
    paused = sum(MON_S for h in hs if (h.get("streams") or {}).get("paused"))
    sp0, sp1 = (h0.get("spill") or {}), (h1.get("spill") or {})
    spill_delta = {k: (sp1.get(k) or 0) - (sp0.get(k) or 0) for k in (
        "written", "loaded", "loaded_tokens", "load_failed", "crc_failed", "write_failed", "no_space", "dropped",
        "unwritten_evicted", "quota_trims", "disk_hits", "disk_lookups")}
    big = [r for r in later if r["miss"] >= BIG_MISS]
    big_busy = [r for r in big if r["phase"] not in ("overload",)]
    resumes = [r["resume"] for r in later]
    returns = [r for r in later if r["kind"] == "return"]
    checks = {
        "no request errors": (not errors, f"{len(errors)} errors"),
        "server stayed up": (not any("health_error" in m for m in mon[-3:]), "health at the end"),
        f"later turns resume >= {MIN_RESUME:.0%} (p10)":
            ((pct(resumes, .1) or 1) >= MIN_RESUME, f"p10 {pct(resumes, .1)}, p50 {pct(resumes, .5)}"),
        f"no {BIG_MISS}+ token re-prefills outside overload":
            (not big_busy, f"{len(big_busy)} ({sum(r['miss'] for r in big_busy)} tokens)"),
        "recall correct": (not any(not r["correct"] and not r.get("truncated") for r in recalls),
                           f"{sum(r['correct'] for r in recalls)}/{len(recalls)}, "
                           f"{sum(bool(r.get('truncated')) for r in recalls)} cut off at {RECALL_TOKENS} tokens"),
        "no spill write/load/crc failures":
            (not any(spill_delta[k] for k in ("load_failed", "crc_failed", "write_failed", "no_space")),
             ", ".join(f"{k} {spill_delta[k]}" for k in ("load_failed", "crc_failed", "write_failed", "no_space"))),
        f"memory stayed above {GUARD_GIB} GiB": (not guard, f"min {min((m['avail_min'] for m in mem.values() if m['avail_min'] is not None), default=None)} GiB"),
        "no 429 outside overload": (not any(v["http_429"] for k, v in per_phase.items() if k != "overload"),
                                    ", ".join(f"{k} {v['http_429']}" for k, v in per_phase.items())),
    }
    warnings = []
    for n, m in mem.items():
        if m["avail_slope_gib_h"] is not None and m["avail_slope_gib_h"] < -1:
            warnings.append(f"{n}: MemAvailable falls {m['avail_slope_gib_h']} GiB/h under load (the decay of 11-12)")
    cut = sum(bool(r.get("truncated")) for r in recalls)
    if cut:
        warnings.append(f"{cut} recalls spent {RECALL_TOKENS} tokens without an answer (reasoning ran long)")
    if paused >= 60:
        warnings.append(f"streams paused ~{paused} s (a request could not be placed in the pool)")
    if kept_max > 0.9 * (h0.get("kept_bytes_cap") or 1 << 62):
        warnings.append(f"kept_bytes reached {kept_max / 2**30:.2f} GiB of {h0['kept_bytes_cap'] / 2**30:.0f}")
    if returns and (pct([r["resume"] for r in returns], .5) or 1) < MIN_RESUME:
        warnings.append(f"after the idle phase, return turns resumed p50 {pct([r['resume'] for r in returns], .5)}")
    return {
        "seconds": round(secs), "requests": len(rows), "turns": len(turns), "recalls": len(recalls),
        "errors": [{k: r.get(k) for k in ("phase", "sid", "turn", "kind", "error")} for r in errors[:20]],
        "phases": per_phase, "memory": mem,
        "kept_bytes_max_gib": round(kept_max / 2**30, 2), "paused_s": paused, "spill": spill_delta,
        "spill_end": {k: sp1.get(k) for k in ("stored_gib", "entries", "cap_bytes", "restore_ms_p95")},
        "big_misses": [{k: r[k] for k in ("phase", "sid", "turn", "prompt", "cached", "miss", "wall")} for r in big],
        "recall_by_depth": [(r["depth"], r["correct"]) for r in recalls],
        "checks": {k: {"ok": ok, "detail": d} for k, (ok, d) in checks.items()},
        "warnings": warnings, "memory_guard": guard,
        "failed": [k for k, (ok, _) in checks.items() if not ok]}


def print_report(s: dict, out: str):
    print(f"\n== long sessions: {s['turns']} turns, {s['recalls']} recalls, {s['seconds'] / 60:.1f} min ==")
    print(f"{'phase':9s} {'turns':>5s} {'prompt tok':>11s} {'reuse':>6s} {'ttft p50/p95':>14s} {'wall p95':>8s}"
          f" {'dec t/s':>7s} {'pre t/s':>7s} {'agg t/s':>7s} {'429':>4s}")
    for name, p in s["phases"].items():
        print(f"{name:9s} {p['turns']:5d} {p['prompt_tokens']:11d} {p['reuse'] or 0:6.1%} "
              f"{p['ttft_p50'] or 0:6.1f}/{p['ttft_p95'] or 0:6.1f} {p['wall_p95'] or 0:8.1f} "
              f"{p['decode_tok_s_p50'] or 0:7.1f} {p['prefill_tok_s_p50'] or 0:7.0f} {p['completion_tok_s'] or 0:7.1f}"
              f" {p['http_429']:4d}")
    print("memory (GiB available start/min/end, slope under load GiB/h; tensorfold RSS max):")
    for n, m in s["memory"].items():
        print(f"  {n:28s} {m['avail_start']}/{m['avail_min']}/{m['avail_end']}  slope {m['avail_slope_gib_h']}"
              f"  rss {m['rss_max']}")
    print(f"kept_bytes max {s['kept_bytes_max_gib']} GiB; spill {s['spill_end']}; spill during run {s['spill']}")
    if s["big_misses"]:
        print(f"big re-prefills ({len(s['big_misses'])}):")
        for b in s["big_misses"][:10]:
            print(f"  {b}")
    for k, v in s["checks"].items():
        print(f"  {'ok  ' if v['ok'] else 'FAIL'} {k}: {v['detail']}")
    for w in s["warnings"]:
        print(f"  warn {w}")
    print(f"files: {out}/turns.jsonl, monitor.jsonl, summary.json")


def parse_phases(spec: str) -> list[tuple[str, int, float]]:
    out = []
    for part in spec.split(","):
        name, count, minutes = part.split(":")
        out.append((name, int(count), float(minutes)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phases", type=parse_phases, default=parse_phases("warm:2:5,steady:8:30,overload:12:10,idle:0:5"))
    ap.add_argument("--grow-to", type=int, default=230_000, help="synthetic sessions compact past this (prod max 229952)")
    ap.add_argument("--time-scale", type=float, default=0.02, help="x production's think time between turns")
    ap.add_argument("--max-gap", type=float, default=120, help="longest wait between a session's turns, seconds")
    ap.add_argument("--replay", help="replay DIR/manifest.json's stored conversations instead (private ids)")
    ap.add_argument("--nodes", type=lambda v: v.split(","), default=nodes_from_local_sh(),
                    help="'local' and ssh hosts to sample memory on (default: here + scripts/local.sh WORKER*)")
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--out", default=time.strftime("long-sessions-%Y%m%d-%H%M%S"))
    args = ap.parse_args()
    print(f"{API_URL}: phases {args.phases}, grow to {args.grow_to}, think time x{args.time_scale}, "
          f"memory on {args.nodes} (guard {GUARD_GIB} GiB), out {args.out}", flush=True)
    return Run(args).go()


if __name__ == "__main__":
    sys.exit(main())
