"""Run every chapter's measurements in one rented L4 session, resumably.

Each step is one measurement subcommand. Its output folder is copied to the results
directory (a mounted Google Drive folder on Colab) the moment the step ends, and a step
whose `.done` marker is already there is skipped, so a disconnected runtime picks up
where it stopped. A failed step is recorded with its exit code and log, never retried
silently, and the run moves on.

    python examples/colab/run_all.py --out /content/drive/MyDrive/aie-results [--only ch03] [--dry-run]
"""
import argparse, json, os, shutil, subprocess, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHAPTERS = [  # (chapter, folder holding measure/run.py), cheapest first
    ("ch03", "ch03-forward-pass-and-kernels"),
    ("ch05", "ch05-model-level-optimization"),
    ("ch06", "ch06-kv-cache"),
    ("ch07", "ch07-batching-and-scheduling"),
]


def steps():
    yield ("ch02", "graphs-pair", [sys.executable, "measure.py", "--serve",
                                   "Qwen/Qwen2.5-7B-Instruct", "--write-baseline",
                                   # greedy, no 1.05 repetition penalty: the setting chapter 3 compares against
                                   "--serve-args=--generation-config vllm"],
           os.path.join(ROOT, "examples", "mini-session"), ["runs", "baseline.json"])
    for ch, folder in CHAPTERS:
        mdir = os.path.join(ROOT, folder, "01_main-chapter-code", "measure")
        if not os.path.exists(os.path.join(mdir, "run.py")):
            yield (ch, "missing", None, mdir, [])
            continue
        r = subprocess.run([sys.executable, "run.py", "--list"], cwd=mdir,
                           capture_output=True, text=True)
        ids = r.stdout.split() if r.returncode == 0 and r.stdout.strip() else ["all"]
        for i in ids:
            yield (ch, i, [sys.executable, "run.py", i], mdir, ["results"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summary = []
    for ch, sid, cmd, cwd, keep in steps():
        if a.only and ch != a.only:
            continue
        dest = os.path.join(a.out, ch, sid)
        if cmd is None:
            print(f"SKIP {ch}: no measure/run.py yet")
            summary.append((ch, sid, "no code"))
            continue
        if os.path.exists(os.path.join(dest, ".done")):
            print(f"DONE already {ch} {sid}")
            summary.append((ch, sid, "done earlier"))
            continue
        print(f"RUN {ch} {sid}: {' '.join(cmd)}", flush=True)
        if a.dry_run:
            summary.append((ch, sid, "dry run"))
            continue
        os.makedirs(dest, exist_ok=True)
        t0 = time.time()
        with open(os.path.join(dest, "log.txt"), "w") as log:
            rc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT).returncode
        for k in keep:  # copy whatever the step produced, success or not
            src = os.path.join(cwd, k)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(dest, k), dirs_exist_ok=True)
            elif os.path.exists(src):
                shutil.copy2(src, dest)
        meta = {"exit": rc, "minutes": round((time.time() - t0) / 60, 1), "cmd": cmd}
        json.dump(meta, open(os.path.join(dest, "step.json"), "w"), indent=1)
        if rc == 0:
            open(os.path.join(dest, ".done"), "w").close()
        summary.append((ch, sid, "ok" if rc == 0 else f"FAILED exit {rc}"))
        print(f"  -> {summary[-1][2]} in {meta['minutes']} min", flush=True)
    print("\nSUMMARY")
    for ch, sid, st in summary:
        print(f"  {ch:5s} {sid:24s} {st}")
    return 1 if any(s.startswith("FAILED") for _, _, s in summary) else 0


if __name__ == "__main__":
    sys.exit(main())
