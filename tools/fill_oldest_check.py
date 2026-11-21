#!/usr/bin/env python3
"""tools/fill_oldest_check.py: patch 0134's TF_GLM_FILL_OLDEST_ROWS, CPU only, in the image (the engine's own
MultiDecoder._group is called on stub lanes; nothing runs on a GPU):

    docker run --rm --network none -e CUDA_VISIBLE_DEVICES= --entrypoint python3 \
      -v "$PWD/tools/fill_oldest_check.py:/c.py:ro" tensorfold-glm53:v0.6.0 /c.py

1. The setting: unset or 0 is off; a multiple of the fill unit up to TF_GLM_FILL_ROWS is taken; anything else is
   refused at start, as TF_GLM_FILL_ROWS is.
2. Off is today's packing: on 3,000 random sets of waiting prompts (lengths, positions, kept points, background
   streams, a busy or idle engine) _group gives exactly what the packing before 0134 gives (a copy of it below).
3. On: the oldest waiting prompt (``first``) leads the chunk and gets min(its rows to its next stop, the share) at
   least; the others follow shortest first; ``first`` takes what they leave; every piece stops at its next kept point
   or end, or a whole fill unit short of it; the chunk never passes its budget; a background stream never passes
   TF_GLM_FILL_ROWS.
4. The case from issue #199: a 150k prompt behind 6 short ones gets 0 rows of a chunk with the setting off, its share
   with it on.
Prints ok / FAIL lines and a total; exit 1 on any FAIL."""
import random
import sys
from types import SimpleNamespace as NS

from tensorfold.families.glm5_next.cuda import multi

fails: list[str] = []
count = 0


def check(name: str, ok: bool) -> None:
    global count
    count += 1
    if not ok:
        fails.append(name)
        print("FAIL " + name)


def lane(sid, n, pos, order, stops=(), background=False, decoding=False):
    return NS(sid=sid, order=order, decoding=decoding, feed=None, head=None, stops=list(stops),
              st=NS(pos=pos), s=NS(prompt=[0] * n, background=background, done=False))


def decoder(lanes, fill=2048, prefill=2048, unit=64, oldest=0):
    return NS(lanes={l.sid: l for l in lanes}, fill_rows=fill, e=NS(prefill_rows=prefill), unit=unit,
              oldest_rows=oldest)


def before_0134(self, first):
    """_group as it was before patch 0134 (patch 0049's packing), for the 'off is unchanged' check."""

    def plain(l):
        return not l.decoding and l.feed is None and l.head is None

    if not plain(first):
        return None
    busy = any(l.decoding and not l.s.done for l in self.lanes.values())
    left = self.fill_rows if busy else self.e.prefill_rows
    lanes = sorted((l for l in self.lanes.values() if plain(l)),
                   key=lambda l: (bool(l.s.background), len(l.s.prompt) - l.st.pos, l.order))
    out = []
    for l in lanes:
        pos, n = l.st.pos, len(l.s.prompt)
        stop = next((p for p in l.stops if p > pos), n)
        room = min(left, self.fill_rows if l.s.background else left)
        if stop - pos > room:
            stop = pos + room // self.unit * self.unit
        if stop <= pos:
            continue
        out.append((l, stop))
        left -= stop - pos
        if left <= 0:
            break
    return out if len(out) > 1 else None


def group(d, first):
    return multi.MultiDecoder._group(d, first)


def pieces(out):
    return None if out is None else [(l.sid, stop) for l, stop in out]


def random_set(rng, unit=64):
    lanes, order = [], 0
    for sid in range(rng.randint(1, 8)):
        n = rng.choice([rng.randint(1, 40) * unit + rng.randint(0, 63), rng.randint(50, 3000) * unit])
        pos = rng.randint(0, max(0, n // unit - 1)) * unit
        stops = sorted({p for p in (rng.randint(pos + 1, n) for _ in range(rng.randint(0, 3))) if p > pos})
        lanes.append(lane(sid, n, pos, order, stops, background=rng.random() < 0.15, decoding=rng.random() < 0.25))
        order += 1
    if all(l.decoding for l in lanes):
        lanes[0].decoding = False
    return lanes


def next_fill(lanes):
    waiting = [l for l in lanes if not l.decoding]
    return next((l for l in waiting if not l.s.background), waiting[0])


# 1. the setting
check("unset is off", multi.fill_oldest_rows(2048, 64, "") == 0)
check("0 is off", multi.fill_oldest_rows(2048, 64, "0") == 0)
check("512 is taken", multi.fill_oldest_rows(2048, 64, "512") == 512)
check("the whole fill chunk is taken", multi.fill_oldest_rows(2048, 64, "2048") == 2048)
for bad in ("100", "4096", "-64", "x", "32"):
    try:
        multi.fill_oldest_rows(2048, 64, bad)
        check(f"{bad!r} refused", False)
    except ValueError:
        check(f"{bad!r} refused", True)
try:
    multi.fill_oldest_rows(2048, 128, "64")
    check("64 refused with a 128-row unit", False)
except ValueError:
    check("64 refused with a 128-row unit", True)

# 2. off is today's packing
rng = random.Random(173)
same = 0
for i in range(3000):
    lanes = random_set(rng)
    d = decoder(lanes, oldest=0)
    first = next_fill(lanes)
    if pieces(group(d, first)) == pieces(before_0134(d, first)):
        same += 1
check(f"off: {same}/3000 random sets packed as before 0134", same == 3000)

# 3. on: the invariants
rng = random.Random(199)
bad = []
for i in range(3000):
    lanes = random_set(rng)
    share = rng.choice([64, 256, 512, 1024, 2048])
    d = decoder(lanes, oldest=share)
    first = next_fill(lanes)
    out = group(d, first)
    busy = any(l.decoding for l in lanes)
    budget = d.fill_rows if busy else d.e.prefill_rows
    if out is None:
        continue
    total = 0
    for l, stop in out:
        pos = l.st.pos
        nxt = next((p for p in l.stops if p > pos), len(l.s.prompt))
        rows = stop - pos
        total += rows
        if not (0 < rows and stop <= nxt and (stop == nxt or rows % d.unit == 0)):
            bad.append(("piece", i))
        if l.s.background and rows > d.fill_rows:
            bad.append(("background", i))
        if l.decoding:
            bad.append(("decoding lane in a chunk", i))
    if total > budget:
        bad.append(("budget", i))
    if out[0][0] is not first:
        bad.append(("first leads", i))
    else:
        pos = first.st.pos
        nxt = next((p for p in first.stops if p > pos), len(first.s.prompt))
        want = min(nxt - pos, share, first.s.background and d.fill_rows or budget)
        want = want if pos + want == nxt else want // d.unit * d.unit
        if out[0][1] - pos < want:
            bad.append(("first's share", i))
    rest = [l for l, _ in out[1:]]
    keys = [(bool(l.s.background), len(l.s.prompt) - l.st.pos, l.order) for l in rest]
    if keys != sorted(keys):
        bad.append(("others shortest first", i))
check(f"on: 3000 random sets keep every invariant ({len(bad)} broken: {bad[:3]})", not bad)

# 4. issue #199's case: a 150k prompt (the oldest) behind six newer short ones, others decoding
waiting = [lane(0, 150_000, 20_480, 0)] + [lane(i, 400 + 64 * i, 0, i) for i in range(1, 7)]
decoding = [lane(10 + i, 5_000, 5_000, 10 + i, decoding=True) for i in range(2)]
for share, want in ((0, 0), (512, 512), (1024, 1024)):
    d = decoder(waiting + decoding, oldest=share)
    out = group(d, waiting[0])
    got = dict(pieces(out) or [(0, waiting[0].st.pos + 2048)])        # None: the oldest fills alone
    rows = got.get(0, 0) - waiting[0].st.pos if 0 in got else 0
    print(f"#199 case, setting {share}: the oldest (150k) prompt gets {rows} of 2048 rows; the chunk "
          f"{pieces(out)}")
    check(f"#199: the 150k prompt gets {rows} rows of a 2048-row chunk with the setting at {share} "
          f"(want at least {want})", rows >= want and (share or rows == 0))

print(f"{count - len(fails)}/{count} ok" + (f"; FAIL: {fails}" if fails else ""))
sys.exit(1 if fails else 0)
