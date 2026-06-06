"""
redo_checkpoint.py  —  Re-run / re-submit checkpoint LUTO runs.

Detects the platform automatically:
  Windows  → NT mode: launches python_script.py as local subprocesses
  Linux    → PBS mode: submits qsub jobs via redo_cmd.sh

A run is classified as:
  • finished    — contains Run_Archive.zip OR Data_RES*.lz4 in output/*/
  • running     — PBS only: its dir is the PBS_O_WORKDIR of a live job
                  NT  only: (skipped — verify manually before running)
  • checkpoint  — has a data_<year>.lz4 inside its output subdir
  • incomplete  — none of the above

Run from the task root directory (the folder that contains Run_G* subdirs).

Common arguments:
    --dry-run        Classify runs and print what would happen; launch/submit nothing.
    --runs   <...>   Specific run names to retry (default: all checkpoint runs).

NT (Windows) only:
    --max    <n>     Max concurrent runs (default: 1).
    --input_dir <p>  Override INPUT_DIR in every run's luto/settings.py.

PBS (Linux) only:
    --mem   <val>    Override PBS memory    (default: inherit from task_param.py).
    --ncpus <val>    Override PBS CPU count (default: inherit from task_param.py).
    --time  <val>    Override PBS walltime  (default: inherit from task_param.py).
    --queue <val>    Override PBS queue     (default: inherit from task_param.py).

Usage:
    python redo_checkpoint.py                        # retry all checkpoints
    python redo_checkpoint.py --dry-run              # preview only
    python redo_checkpoint.py --runs Run_G0001       # specific run only
    python redo_checkpoint.py --max 2                # NT: 2 at a time
    python redo_checkpoint.py --time 24:00:00        # PBS: override walltime
"""

import argparse
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path


HERE       = Path(__file__).parent
IS_WINDOWS = platform.system() == "Windows"


# ════════════════════════════════════════════════════════════════════════════
# Common helpers
# ════════════════════════════════════════════════════════════════════════════

def get_output_subdirs(run_dir: Path) -> list[Path]:
    """Return all subdirectories of run_dir/output/, sorted by name."""
    output_dir = run_dir / "output"
    if not output_dir.is_dir():
        return []
    return sorted([d for d in output_dir.iterdir() if d.is_dir()], key=lambda d: d.name)


def get_checkpoint_lz4(run_dir: Path) -> list[Path]:
    """Return sorted data_<year>.lz4 checkpoint files inside run_dir/output/*/."""
    output_dir = run_dir / "output"
    if not output_dir.is_dir():
        return []
    return sorted(
        f
        for d in output_dir.iterdir() if d.is_dir()
        for f in d.iterdir() if re.match(r"data_\d{4}\.lz4", f.name)
    )


def is_finished(run_dir: Path) -> bool:
    """Finished = Run_Archive.zip exists OR Data_RES*.lz4 exists in output/*/."""
    if (run_dir / "Run_Archive.zip").exists():
        return True
    output_dir = run_dir / "output"
    if not output_dir.is_dir():
        return False
    return any(
        re.match(r"Data_RES\d+\.lz4", f.name)
        for d in output_dir.iterdir() if d.is_dir()
        for f in d.iterdir()
    )


def classify_runs(
    root: Path,
    requested: list[str],
    running_dirs: set[Path] | None = None,
) -> dict[str, list[Path]]:
    """
    Classify every Run_G* directory under root.

    running_dirs: PBS only — set of live job workdirs from qstat.
                  Pass None (NT mode) to skip running detection.
    """
    finished, running, checkpoint, incomplete = [], [], [], []

    candidates = (
        [root / n for n in requested] if requested
        else sorted(root.glob("Run_G*"))
    )

    for run_dir in candidates:
        if not run_dir.is_dir():
            print(f"[warn] {run_dir.name} not found — skipped")
            continue
        if is_finished(run_dir):
            finished.append(run_dir)
        elif running_dirs is not None and run_dir.resolve() in {d.resolve() for d in running_dirs}:
            running.append(run_dir)
        elif get_checkpoint_lz4(run_dir):
            checkpoint.append(run_dir)
        else:
            incomplete.append(run_dir)

    return dict(finished=finished, running=running, checkpoint=checkpoint, incomplete=incomplete)


def print_classification(runs: dict[str, list[Path]]) -> None:
    print(f"Finished   ({len(runs['finished'])}): {[d.name for d in runs['finished']]}")
    if runs["running"]:
        print(f"Running    ({len(runs['running'])}):  {[d.name for d in runs['running']]}")
    print(f"Checkpoint ({len(runs['checkpoint'])}): {[d.name for d in runs['checkpoint']]}")
    print(f"Incomplete ({len(runs['incomplete'])}): {[d.name for d in runs['incomplete']]}")

    if runs["checkpoint"]:
        print()
        for d in runs["checkpoint"]:
            lz4s = get_checkpoint_lz4(d)
            yr   = int(lz4s[-1].stem.split("_")[1]) if lz4s else "?"
            print(f"  {d.name}: resume from year {yr}  ({lz4s[-1].name if lz4s else '—'})")


def filter_multi_output(dirs: list[Path], dry_run: bool) -> list[Path]:
    """
    Warn and skip any run that has more than one output subdir.
    Returns the subset with exactly one output dir.
    """
    clean, skipped = [], []
    for d in dirs:
        out_subs = get_output_subdirs(d)
        if len(out_subs) > 1:
            skipped.append(d)
            tag = "[dry-run] " if dry_run else ""
            print(f"  {tag}[skip] {d.name}: {len(out_subs)} output dirs — keep only the one to resume:")
            for s in out_subs:
                print(f"    {s.name}")
        else:
            clean.append(d)
    if skipped:
        print(f"\n  {len(skipped)} run(s) skipped due to multiple output dirs.\n")
    return clean


# ════════════════════════════════════════════════════════════════════════════
# PBS (Linux) — submit via qsub
# ════════════════════════════════════════════════════════════════════════════

def get_running_workdirs() -> set[Path]:
    """Return PBS_O_WORKDIR paths for all live jobs owned by $USER."""
    user = os.environ.get("USER", "")
    id_result = subprocess.run(["qstat", "-u", user], capture_output=True, text=True)
    job_ids = [
        line.split()[0]
        for line in id_result.stdout.splitlines()
        if line and "." in line.split()[0]
        and not line.startswith("-") and not line.startswith(" ")
    ]

    dirs: set[Path] = set()
    for job_id in job_ids:
        detail = subprocess.run(["qstat", "-f", job_id], capture_output=True, text=True)
        joined: list[str] = []
        for raw in detail.stdout.splitlines():
            if raw.startswith("\t") and joined:
                joined[-1] += raw.strip()
            else:
                joined.append(raw.strip())
        for line in joined:
            if "PBS_O_WORKDIR=" not in line:
                continue
            for token in line.split(","):
                token = token.strip()
                if token.startswith("PBS_O_WORKDIR="):
                    dirs.add(Path(token.split("=", 1)[1].strip()))
    return dirs


def write_redo_param(run_dir: Path, job_name: str, args: argparse.Namespace) -> None:
    existing = {}
    task_param = run_dir / "task_param.py"
    if task_param.exists():
        for line in task_param.read_text().splitlines():
            m = re.match(r'export (\w+)="([^"]*)"', line)
            if m:
                existing[m.group(1)] = m.group(2)

    mem   = args.mem   or existing.get("MEM",   "320gb")
    ncpus = args.ncpus or existing.get("NCPUS", "64")
    time_ = args.time  or existing.get("TIME",  "48:00:00")
    queue = args.queue or existing.get("QUEUE", "normalsr")

    lines = [
        f'export MEM="{mem}"\n',
        f'export NCPUS="{ncpus}"\n',
        f'export TIME="{time_}"\n',
        f'export QUEUE="{queue}"\n',
        f'export JOB_NAME="{job_name}"\n',
    ]
    (run_dir / "redo_param.py").write_text("".join(lines))
    print(f"    wrote redo_param.py  (mem={mem}, ncpus={ncpus}, time={time_}, queue={queue}, job={job_name})")


def submit_job_pbs(run_dir: Path, args: argparse.Namespace) -> None:
    out_subs = get_output_subdirs(run_dir)
    if len(out_subs) > 1:
        print(f"  [skip] {run_dir.name}: {len(out_subs)} output dirs — keep only the one to resume:")
        for s in out_subs:
            print(f"    {s.name}")
        return

    lz4_files = get_checkpoint_lz4(run_dir)
    if not lz4_files:
        print(f"  [skip] {run_dir.name}: no checkpoint lz4 found (unexpected)")
        return

    yr = int(lz4_files[-1].stem.split("_")[1])
    print(f"  {run_dir.name}:")
    print(f"    checkpoint year : {yr}  ({lz4_files[-1].name})")
    print(f"    output dir      : {out_subs[0].name if out_subs else '—'}")

    if args.dry_run:
        print(f"    [dry-run] skipping qsub")
        return

    write_redo_param(run_dir, job_name=f"redo_{run_dir.name}", args=args)

    result = subprocess.run(
        ["bash", "redo_cmd.sh"],
        cwd=run_dir,
        capture_output=True, text=True,
    )
    job_id = result.stdout.strip()
    if result.returncode == 0 and job_id:
        print(f"    [submitted] {job_id}")
    else:
        err = result.stderr.strip() or result.stdout.strip() or "(no output)"
        print(f"    [error] {err}")


def main_pbs(args: argparse.Namespace, root: Path) -> None:
    print("Mode: PBS (Linux)\n")
    print("Querying running PBS jobs …")
    running_dirs = get_running_workdirs()

    runs = classify_runs(root, args.runs, running_dirs=running_dirs)
    print_classification(runs)

    if not runs["checkpoint"]:
        print("\nNo checkpoint runs to resubmit.")
        return

    print(f"\nSubmitting {len(runs['checkpoint'])} checkpoint job(s) …\n")
    for run_dir in runs["checkpoint"]:
        submit_job_pbs(run_dir, args)


# ════════════════════════════════════════════════════════════════════════════
# NT (Windows) — launch as local subprocesses
# ════════════════════════════════════════════════════════════════════════════

def launch_run_nt(run_dir: Path, input_dir: str | None) -> tuple[subprocess.Popen, object]:
    """Start python_script.py in run_dir; log stdout+stderr to redo.log."""
    log_path = run_dir / "redo.log"
    log_file = open(log_path, "w", encoding="utf-8")

    cmd = [sys.executable, "python_script.py"]
    if input_dir:
        cmd += ["--input_dir", input_dir]

    proc = subprocess.Popen(
        cmd,
        cwd=run_dir,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    print(f"  [start]  {run_dir.name}  pid={proc.pid}  → {log_path}")
    return proc, log_file


def run_all_nt(
    checkpoint_dirs: list[Path],
    max_concurrent: int,
    input_dir: str | None,
    dry_run: bool,
) -> None:
    target = filter_multi_output(checkpoint_dirs, dry_run)

    if dry_run:
        print(f"\n[dry-run] Would retry {len(target)} run(s) (max {max_concurrent} at a time):")
        for d in target:
            lz4s = get_checkpoint_lz4(d)
            yr   = int(lz4s[-1].stem.split("_")[1]) if lz4s else "?"
            print(f"  {d.name}  checkpoint year={yr}  ({lz4s[-1].name if lz4s else '—'})")
        return

    queue  = list(target)
    active: dict[str, tuple[subprocess.Popen, object, Path]] = {}
    done, failed = [], []

    print(f"\nRetrying {len(queue)} checkpoint run(s), max_concurrent={max_concurrent}\n")

    while queue or active:
        while queue and len(active) < max_concurrent:
            run_dir        = queue.pop(0)
            proc, log_file = launch_run_nt(run_dir, input_dir)
            active[run_dir.name] = (proc, log_file, run_dir)

        for name in list(active):
            proc, log_file, run_dir = active[name]
            if proc.poll() is not None:
                log_file.close()
                if proc.returncode == 0:
                    done.append(name)
                    print(f"  [done]   {name}  ✓")
                else:
                    failed.append(name)
                    print(f"  [failed] {name}  ✗  exit={proc.returncode}  see {run_dir / 'redo.log'}")
                del active[name]

        if active:
            time.sleep(10)

    print(f"\n--- Summary ---")
    print(f"Done   ({len(done)}): {done}")
    if failed:
        print(f"Failed ({len(failed)}): {failed}")


def main_nt(args: argparse.Namespace, root: Path) -> None:
    print("Mode: NT (Windows)\n")
    print("(Running-process check skipped — verify manually before launching.)\n")

    runs = classify_runs(root, args.runs, running_dirs=None)
    print_classification(runs)

    if not runs["checkpoint"]:
        print("\nNo checkpoint runs to retry.")
        return

    target = runs["checkpoint"]
    if args.runs:
        requested_names = set(args.runs)
        target = [d for d in target if d.name in requested_names]
        if not target:
            print("\nNone of the requested runs have checkpoints.")
            return

    run_all_nt(target, args.max, args.input_dir, args.dry_run)


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Re-run/re-submit checkpoint LUTO runs (auto-detects Windows vs PBS)."
    )
    # Common
    parser.add_argument("--dry-run", action="store_true", help="Preview only — launch/submit nothing")
    parser.add_argument("--runs",    nargs="*", default=[], help="Specific run names (default: all checkpoints)")
    # NT only
    parser.add_argument("--max",       type=int, default=1, help="[NT] Max concurrent runs (default: 1)")
    parser.add_argument("--input_dir", default=None,        help="[NT] Override INPUT_DIR in every run's settings.py")
    # PBS only
    parser.add_argument("--mem",   default=None, help="[PBS] Override PBS memory")
    parser.add_argument("--ncpus", default=None, help="[PBS] Override PBS CPU count")
    parser.add_argument("--time",  default=None, help="[PBS] Override PBS walltime")
    parser.add_argument("--queue", default=None, help="[PBS] Override PBS queue")
    args = parser.parse_args()

    root = Path.cwd()
    print(f"Root: {root}")

    if IS_WINDOWS:
        main_nt(args, root)
    else:
        main_pbs(args, root)


if __name__ == "__main__":
    main()
