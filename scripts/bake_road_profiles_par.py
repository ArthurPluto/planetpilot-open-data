"""Parallel road profile pre-bake: re-bakes only the stale areas of
data/world/odbl/roadprof/ (docs/ROAD_NETWORK.md section 6) in several headless
Godot processes and merges the index.

    python tools/bake_road_profiles_par.py [--jobs 4] [--all] [--areas n30e130,n40e000] [--dry-run]
        [--job-edges 40000] [--out DIR] [--userdir DIR] [--queue C:/tmp/sbq/heavy.sh|none] [--verify]

1. keys: one Godot run (tools/bake_road_profiles.gd keys=...) writes every area's
   current key (RoadBuild.area_key: global part, graph, packs, airports, hulls,
   stamps, city water within the area's box).
2. stale = areas whose key differs from index.json, or with no entry or file
   (--all: every area; --areas: these too). Areas no longer in the graph are
   dropped with their files. Each stale area is printed with the key parts that
   changed.
3. The stale areas are packed into jobs of about --job-edges edges (largest
   first, onto the lightest job; at least --jobs jobs) and run --jobs at a time,
   each a Godot process `bake_road_profiles.gd areas=... index=0` with its own
   APPDATA (--userdir/j<n>), through the shared queue (--queue: bash <queue> cpu ...;
   none: run directly). A job's time is measured from when it got its slot.
4. merge: index.json gets the new entries ("AREA {json}" lines of the jobs); an
   area whose job failed keeps its old entry (still stale: the game solves it live).
--verify: a second keys run afterwards must find no stale area.
Never writes outside --out, --userdir and the log folder (--userdir/logs).
"""
import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from godot_bin import GODOT  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "world", "odbl", "roadprof")
QUEUE = "C:/tmp/sbq/heavy.sh"
TOOL = "res://tools/bake_road_profiles.gd"


def find_bash():
    """Git Bash (on Windows a bare "bash" may be WSL's, which cannot run the queue)."""
    for b in (os.environ.get("SKYBOUND_BASH", ""), r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
        if b and os.path.exists(b):
            return b
    return shutil.which("bash") or "bash"


BASH = find_bash()
VERSION = 2


def godot_cmd(args, queue):
    cmd = [GODOT, "--headless", "--audio-driver", "Dummy", "--path", ROOT, "-s", TOOL, "--"] + args
    # the START line marks when the queue gave the job its slot
    inner = 'echo "START $(date +%s.%N)"; exec "$@"'
    if queue and queue != "none":
        return [BASH, queue, "cpu", "bash", "-c", inner, "job"] + cmd
    return [BASH, "-c", inner, "job"] + cmd


def run(args, queue, appdata, log):
    os.makedirs(appdata, exist_ok=True)
    env = dict(os.environ)
    env["APPDATA"] = appdata
    t0 = time.time()
    with open(log, "w", encoding="utf-8", newline="\n") as f:
        p = subprocess.run(godot_cmd(args, queue), stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    t1 = time.time()
    start = t0
    areas = {}
    with open(log, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("START "):
                try:
                    start = float(line.split()[1])
                except ValueError:
                    pass
            elif line.startswith("AREA "):
                r = json.loads(line[5:])
                areas[r.pop("name")] = r
    return {"rc": p.returncode, "areas": areas, "wait": max(0.0, start - t0), "run": t1 - max(start, t0), "log": log}


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_index(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(json.dumps(doc, indent="\t", sort_keys=True) + "\n")
    os.replace(tmp, path)


def get_keys(a, queue, tag):
    kp = os.path.join(a.userdir, "keys_%s.json" % tag)
    if os.path.exists(kp):
        os.remove(kp)
    r = run(["keys=" + kp.replace("\\", "/")], queue, os.path.join(a.userdir, "keys"), os.path.join(a.logs, "keys_%s.log" % tag))
    keys = load_json(kp)
    if r["rc"] != 0 or not keys:
        sys.exit("keys run failed (rc %d), see %s" % (r["rc"], r["log"]))
    print("keys: %d areas, global %d (%.0f s run, %.0f s queued)" % (len(keys["areas"]), keys["global"], r["run"], r["wait"]))
    return keys


def stale_areas(keys, index, out, a):
    names = keys["parts"]
    old = (index or {}).get("areas", {}) if (index or {}).get("version") == VERSION else {}
    want = {}
    for n, k in sorted(keys["areas"].items()):
        o = old.get(n)
        why = None
        if a.all:
            why = "all"
        elif n in a.only:
            why = "asked"
        elif o is None:
            why = "new area"
        elif not os.path.exists(os.path.join(out, n + ".bin")):
            why = "file missing"
        elif int(o.get("key", -1)) != int(k["key"]):
            op = o.get("parts", [])
            why = ", ".join(names[i] for i in range(len(names)) if i >= len(op) or int(op[i]) != int(k["parts"][i])) or "key"
        if why:
            want[n] = why
    gone = sorted(n for n in old if n not in keys["areas"])
    return want, gone


def pack(want, keys, njobs, job_edges):
    sizes = sorted(((keys["areas"][n]["edges"], n) for n in want), reverse=True)
    total = sum(s for s, _ in sizes)
    n = max(1, min(len(sizes), max(njobs, -(-total // max(1, job_edges)))))
    jobs = [[0, []] for _ in range(n)]
    for s, name in sizes:
        j = min(jobs, key=lambda x: x[0])
        j[0] += s
        j[1].append(name)
    return [j for j in jobs if j[1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=4, help="Godot processes at once (default 4: the queue's cpu slots)")
    ap.add_argument("--all", action="store_true", help="re-bake every area")
    ap.add_argument("--areas", default="", help="also re-bake these areas (comma list of area names)")
    ap.add_argument("--job-edges", type=int, default=40000, help="edges per job (about 7 ms each: 40,000 is about 5 min)")
    ap.add_argument("--out", default=OUT, help="bake folder (index.json + <area>.bin)")
    ap.add_argument("--userdir", default=os.path.join(tempfile.gettempdir(), "roadprof_par"),
                    help="APPDATA root of the Godot runs (j<n> per job) and the logs")
    ap.add_argument("--queue", default=QUEUE, help="slot queue script (bash <queue> cpu ...), none: run directly")
    ap.add_argument("--dry-run", action="store_true", help="list the stale areas and the jobs, bake nothing")
    ap.add_argument("--verify", action="store_true", help="a second keys run must find nothing stale")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # progress shows in a redirected log too
    a.only = set(x.strip() for x in a.areas.split(",") if x.strip())
    a.out = os.path.abspath(a.out)
    a.logs = os.path.join(a.userdir, "logs")
    os.makedirs(a.logs, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    keys = get_keys(a, a.queue, "before")
    ipath = os.path.join(a.out, "index.json")
    index = load_json(ipath)
    want, gone = stale_areas(keys, index, a.out, a)
    fresh = len(keys["areas"]) - len(want)
    print("%d areas: %d fresh, %d to bake, %d gone" % (len(keys["areas"]), fresh, len(want), len(gone)))
    for n in sorted(want):
        print("  %-8s %7d edges  %s" % (n, keys["areas"][n]["edges"], want[n]))
    jobs = pack(want, keys, a.jobs, a.job_edges)
    for i, (s, names) in enumerate(jobs):
        print("  job %d: %d edges, %s" % (i, s, ",".join(sorted(names))))
    if a.dry_run:
        return
    results = {}
    t_bake = time.time()
    with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        futs = {}
        for i, (s, names) in enumerate(jobs):
            args = ["areas=" + ",".join(sorted(names)), "index=0", "out=" + a.out.replace("\\", "/")]
            futs[ex.submit(run, args, a.queue, os.path.join(a.userdir, "j%d" % i), os.path.join(a.logs, "job%d.log" % i))] = (i, s, names)
        for f in cf.as_completed(futs):
            i, s, names = futs[f]
            r = f.result()
            results[i] = r
            print("job %d done: rc %d, %d of %d areas, %.0f s run, %.0f s queued (%s)" % (
                i, r["rc"], len(r["areas"]), len(names), r["run"], r["wait"], r["log"]))
    bake_wall = time.time() - t_bake
    # merge
    doc = index if index and index.get("version") == VERSION else {}
    areas = dict(doc.get("areas", {}))
    for n in gone:
        areas.pop(n, None)
        p = os.path.join(a.out, n + ".bin")
        if os.path.exists(p):
            os.remove(p)
    done = {}
    for r in results.values():
        done.update(r["areas"])
    failed = sorted(n for n in want if n not in done)
    for n, r in done.items():
        areas[n] = r
    doc = {"version": VERSION, "area_deg": keys["area_deg"], "margin_deg": keys["margin_deg"], "global": keys["global"],
           "parts": keys["parts"], "areas": dict(sorted(areas.items()))}
    write_index(ipath, doc)
    # report
    print("\narea      records     MB      km   min")
    for n in sorted(done, key=lambda x: -done[x]["ms"]):
        r = done[n]
        print("%-8s %8d %6.2f %7.0f %5.1f" % (n, r["records"], r["bytes"] / 1e6, r["km"], r["ms"] / 60000.0))
    sizes = [os.path.getsize(os.path.join(a.out, n + ".bin")) for n in areas if os.path.exists(os.path.join(a.out, n + ".bin"))]
    cpu = sum(r["ms"] for r in done.values()) / 60000.0
    runs = [r["run"] for r in results.values()]
    print("\n%d areas baked, %d failed%s; %d files, %.1f MB, largest %.2f MB" % (
        len(done), len(failed), (": " + ",".join(failed)) if failed else "", len(sizes), sum(sizes) / 1e6, max(sizes or [0]) / 1e6))
    print("bake: %.1f min wall (jobs %.1f to %.1f min each), %.1f min of area CPU; total with keys %.1f min" % (
        bake_wall / 60.0, min(runs or [0]) / 60.0, max(runs or [0]) / 60.0, cpu, (time.time() - t_start) / 60.0))
    if a.verify:
        k2 = get_keys(a, a.queue, "after")
        w2, _ = stale_areas(k2, load_json(ipath), a.out, argparse.Namespace(all=False, only=set()))
        print("verify: %d stale after the bake%s" % (len(w2), (": " + ",".join(sorted(w2))) if w2 else ""))
        if w2:
            sys.exit(1)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
