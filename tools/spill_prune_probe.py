#!/usr/bin/env python3
"""When does an idle ~200k-token conversation leave the spill store, and what does resuming it cost then?

Usage: tools/spill_prune_probe.py [--targets 6] [--size 200000] [--sessions 8] [--max-minutes 120] [--out DIR]
       (API_URL / PORT as in client.py; run on a server nothing else uses, on rank 0's machine)

The spill tier frees room in this order (patch 0128): old turns, abandoned forks, whole conversations least recently
used first (a conversation's largest states first), shared system prompts last. So how long an idle conversation stays
depends on how much other traffic is written after its last use, and a request to it would reset that, so it is never
asked while the run watches. Instead:

  1. build --targets conversations of ~--size tokens one after the other on the idle server, each behind its own system
     prompt, in three turns (~11k, ~106k, ~201k) like an agent's, and note the spill keys each one adds (rank 0's
     points/<key>.json, the key ending in the state's token count);
  2. run --sessions synthetic agent sessions (tools/long_sessions.py's) as pressure, and every 15 s list the store's
     keys: when a target's key goes, log the time since the target's last use and the GiB the other sessions wrote;
  3. stop when every target's largest state is gone (or --max-minutes, or the memory guard), stop the pressure, and
     resume each target once: the tokens it still finds (cached) and the time to its first token.

The spill folder comes from the container's /spill mount (CONTAINER, default glm53-flash-tf), the build's subfolder
from /health. Output in --out: events.jsonl, monitor.jsonl, summary.json; a report on stdout. Exit code 2 when the
server is not there, 3 when the memory guard stopped the run.
"""
import argparse
import glob
import json
import os
import random
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import long_sessions as L  # noqa: E402

CONTAINER = os.environ.get("CONTAINER", "glm53-flash-tf")
WATCH_S = 15


def spill_points_dir() -> str:
    src = subprocess.run(["docker", "inspect", CONTAINER, "--format",
                          '{{range .Mounts}}{{if eq .Destination "/spill"}}{{.Source}}{{end}}{{end}}'],
                         capture_output=True, text=True, check=True).stdout.strip()
    builds = sorted(glob.glob(os.path.join(src, "spill-*", "rank0", "points")), key=os.path.getmtime)
    if not src or not builds:
        sys.exit(f"no spill store under {CONTAINER}'s /spill mount ({src!r})")
    return builds[-1]                                         # the live build's folder is the one written last


def keys(points: str) -> set[str]:
    return {f[:-5] for f in os.listdir(points) if f.endswith(".json")}


def tokens_of(key: str) -> int:
    return int(key.rsplit("-", 1)[1])


def wait_spill_idle(timeout=300):
    """Until the spill tier has written what it keeps (its pending count 0)."""
    time.sleep(3)                                             # the keep's write is queued just after the reply ends
    end = time.time() + timeout
    while time.time() < end:
        if not (L.health().get("spill") or {}).get("pending"):
            return
        time.sleep(2)


def written_bytes() -> int:
    return (L.health().get("spill") or {}).get("written_total_bytes") or 0


class Target(L.Synthetic):
    """A conversation behind its own system prompt, grown to ~size tokens in three turns."""

    def __init__(self, sid, size):
        super().__init__(sid, random.Random(sid), grow_to=10**9)
        self.size = size
        self.system = L.text_of(L.SYSTEM_TOKENS, 90_000 + sid)  # its own: not a prefix the pressure shares
        self._fresh(opening=L.text_of(200, 91_000 + sid))
        self.keys: set[str] = set()

    def grow(self, add: int):
        self.messages.append({"role": "user", "content": "Tool output:\n" + L.text_of(add, 92_000 + self.sid * 10
                                                                                       + len(self.messages))})


def build(targets, size, points, log):
    step = (size - L.SYSTEM_TOKENS) // 2
    for t in targets:
        before = keys(points)
        for i in range(3):
            path, body = t.request()
            body["max_tokens"] = 64
            r = L.stream(path, body)
            t.messages.append({"role": "assistant", "content": r["text"] or "ok"})
            if i < 2:
                t.grow(step)
            t.prompt = r["prompt"]
        wait_spill_idle()
        t.keys = keys(points) - before
        t.last_used = time.time()
        log({"event": "built", "target": t.sid, "prompt": t.prompt, "keys": sorted(t.keys, key=tokens_of),
             "secs": round(time.time() - t.last_used, 1)})
        print(f"  target {t.sid}: {t.prompt} tokens, spill keys {sorted(tokens_of(k) for k in t.keys)}", flush=True)


def pressure(n, stop, scale, max_gap, log):
    rng = random.Random(4242)
    sessions = [L.Synthetic(100 + i, random.Random(rng.random()), 230_000) for i in range(n)]

    def loop(s):
        while not stop.is_set():
            path, body = s.request()
            try:
                r = L.stream(path, body)
            except L.Busy:
                stop.wait(3)
                continue
            except Exception as exc:                          # noqa: BLE001
                log({"event": "pressure_error", "sid": s.sid, "error": str(exc)[:200]})
                stop.wait(5)
                continue
            s.prev_prompt = r["prompt"]
            s.turns += 1
            s.after(r)
            stop.wait(min(s.gap(scale), max_gap))

    threads = [threading.Thread(target=loop, args=(s,), daemon=True) for s in sessions]
    for t in threads:
        t.start()
    return threads


def probe(t: Target) -> dict:
    t.messages.append({"role": "user", "content": "Summarize the last tool output in one sentence."})
    path, body = t.request()
    body["max_tokens"] = 32
    r = L.stream(path, body)
    return {"prompt": r["prompt"], "cached": r["cached"], "ttft": round(r["ttft"] or 0, 2),
            "prefill_s": r["prefill_s"], "wall": round(r["wall"], 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--targets", type=int, default=6)
    ap.add_argument("--size", type=int, default=200_000)
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--max-minutes", type=float, default=120)
    ap.add_argument("--time-scale", type=float, default=0.02)
    ap.add_argument("--max-gap", type=float, default=120)
    ap.add_argument("--nodes", type=lambda v: v.split(","), default=L.nodes_from_local_sh())
    ap.add_argument("--out", default=time.strftime("spill-prune-%Y%m%d-%H%M%S"))
    a = ap.parse_args()
    try:
        h0 = L.health()
    except Exception as exc:                                  # noqa: BLE001
        print(f"cannot reach {L.API_URL}/health ({exc})")
        return 2
    points = spill_points_dir()
    os.makedirs(a.out, exist_ok=True)
    ev = open(os.path.join(a.out, "events.jsonl"), "w")

    def log(row):
        row.setdefault("t", time.time())
        ev.write(json.dumps(row) + "\n")
        ev.flush()

    sp = h0.get("spill") or {}
    print(f"{L.API_URL}: spill {points}, {sp.get('stored_gib')} of {(sp.get('cap_bytes') or 0) / 2**30:.0f} GiB, "
          f"{sp.get('entries')} entries; {a.targets} targets of ~{a.size} tokens, {a.sessions} pressure sessions",
          flush=True)
    log({"event": "start", "spill": sp})
    stop = threading.Event()
    mon = L.Monitor(open(os.path.join(a.out, "monitor.jsonl"), "w"), a.nodes, stop)
    mon.phase = "build"
    mon.start()
    targets = [Target(i, a.size) for i in range(a.targets)]
    gone: dict[str, dict] = {}
    workers: list[threading.Thread] = []
    try:
        print("== build (idle server, one target at a time)", flush=True)
        build(targets, a.size, points, log)
        mon.phase = "pressure"
        w0, t0 = written_bytes(), time.time()
        print(f"== pressure: {a.sessions} sessions until every target's largest state is gone "
              f"(at most {a.max_minutes} min)", flush=True)
        workers = pressure(a.sessions, stop, a.time_scale, a.max_gap, log)
        while not stop.is_set() and time.time() - t0 < a.max_minutes * 60:
            now, present = time.time(), keys(points)
            h = L.health()
            s = h.get("spill") or {}
            for t in targets:
                for k in t.keys - present:
                    if k in gone:
                        continue
                    gone[k] = {"event": "pruned", "target": t.sid, "key_tokens": tokens_of(k),
                               "largest": tokens_of(k) == max(map(tokens_of, t.keys)),
                               "idle_min": round((now - t.last_used) / 60, 1),
                               "pressure_min": round((now - t0) / 60, 1),
                               "others_wrote_gib": round(((s.get("written_total_bytes") or 0) - w0) / 2**30, 2),
                               "stored_gib": s.get("stored_gib"), "entries": s.get("entries"),
                               "pruned": {x: s.get(x) for x in ("pruned_turns", "pruned_forks",
                                                                 "pruned_conversations", "pruned_shared")}}
                    log(gone[k])
                    g = gone[k]
                    print(f"  pruned: target {t.sid}'s {g['key_tokens']}-token state"
                          f"{' (its largest)' if g['largest'] else ''} after {g['idle_min']} min idle, "
                          f"{g['others_wrote_gib']} GiB written by others", flush=True)
            log({"event": "tick", "pressure_min": round((now - t0) / 60, 2), "present": {
                t.sid: sorted(tokens_of(k) for k in t.keys & present) for t in targets},
                "written_gib": round(((s.get("written_total_bytes") or 0) - w0) / 2**30, 2),
                "stored_gib": s.get("stored_gib"), "entries": s.get("entries")})
            if all(max(t.keys, key=tokens_of) in gone for t in targets if t.keys):
                print("  every target's largest state is gone", flush=True)
                break
            stop.wait(WATCH_S)
        guard = mon.guard_tripped
        pressure_min = round((time.time() - t0) / 60, 1)
        written = round((written_bytes() - w0) / 2**30, 2)
    finally:
        stop.set()
        mon.close()
    for w in workers:
        w.join(timeout=900)
    if guard:
        print(f"memory guard: {guard}")
        return 3
    print("== resume each target once (pressure stopped)", flush=True)
    wait_spill_idle()
    probes = {}
    for t in targets:
        p = probe(t)
        p.update(target=t.sid, kept_keys=sorted(tokens_of(k) for k in t.keys if k not in gone))
        probes[t.sid] = p
        log({"event": "probe", **p})
        print(f"  target {t.sid}: {p['prompt']} tokens, cached {p['cached']} "
              f"({p['cached'] / p['prompt']:.0%}), ttft {p['ttft']} s, server prefill {p['prefill_s']} s", flush=True)
    summary = {"pressure_min": pressure_min, "others_wrote_gib": written, "spill_start": sp,
               "spill_end": L.health().get("spill"), "pruned": list(gone.values()), "probes": probes,
               "targets": [{"target": t.sid, "prompt": t.prompt, "keys": sorted(map(tokens_of, t.keys))}
                           for t in targets]}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(f"\n== {pressure_min} min of {a.sessions}-session pressure, {written} GiB written by others")
    for t in targets:
        big = max(t.keys, key=tokens_of) if t.keys else None
        g = gone.get(big)
        p = probes[t.sid]
        print(f"  target {t.sid}: largest state "
              + (f"pruned after {g['idle_min']} min idle / {g['others_wrote_gib']} GiB" if g else "still stored")
              + f"; resume found {p['cached']} of {p['prompt']} tokens in {p['ttft']} s")
    print(f"files: {a.out}/events.jsonl, monitor.jsonl, summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
