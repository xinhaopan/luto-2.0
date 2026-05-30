"""
redo_checkpoint.py  —  Re-submit PBS jobs for runs that stalled mid-simulation.

A run is classified as:
  • finished   — contains Run_Archive.zip
  • running    — its directory is the PBS_O_WORKDIR of a live Gadi job
  • checkpoint — has a data_<year>.lz4 inside its output subdir
  • incomplete — none of the above

For each checkpoint run the script writes redo_param.py (updated PBS settings)
and calls `bash redo_cmd.sh` from the run dir. python_script.py detects the
existing checkpoint, skips load_data(), and resumes from the saved year.

Usage:
    python redo_checkpoint.py                    # inherit PBS settings from each run's task_param.py
    python redo_checkpoint.py --dry-run          # print what would happen; no files written, no qsub
    python redo_checkpoint.py --time 24:00:00    # override one param; rest inherited from task_param.py
"""

import argparse
import os
import re
import subprocess
from pathlib import Path


def get_running_workdirs() -> set[Path]:
    """Return PBS_O_WORKDIR paths for all live jobs owned by $USER."""
    result = subprocess.run(
        ["qstat", "-f", "-u", os.environ.get("USER", "")],
        capture_output=True, text=True, check=True,
    )

    dirs: set[Path] = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if "PBS_O_WORKDIR=" in line:
            for token in line.split(","):
                token = token.strip()
                if token.startswith("PBS_O_WORKDIR="):
                    dirs.add(Path(token.split("=", 1)[1].strip()))
    return dirs


def classify_runs(root: Path, running_dirs: set[Path]) -> dict[str, list[Path]]:
    finished, running, checkpoint, incomplete = [], [], [], []
    for run_dir in sorted(root.glob("Run_G*")):
        if not run_dir.is_dir():
            continue
        if (run_dir / "Run_Archive.zip").exists():
            finished.append(run_dir)
        elif run_dir.resolve() in {d.resolve() for d in running_dirs}:
            running.append(run_dir)
        elif any(re.match(r'data_\d{4}\.lz4', f.name) for d in (run_dir / 'output').iterdir() if d.is_dir() for f in d.iterdir()):
            checkpoint.append(run_dir)
        else:
            incomplete.append(run_dir)
    return dict(finished=finished, running=running, checkpoint=checkpoint, incomplete=incomplete)


def write_redo_param(run_dir: Path, job_name: str, args: argparse.Namespace) -> None:
    """Write redo_param.py, inheriting task_param.py values for any unspecified args."""
    existing = {}
    task_param = run_dir / "task_param.py"
    if task_param.exists():
        for line in task_param.read_text().splitlines():
            m = re.match(r'export (\w+)="([^"]*)"', line)
            if m:
                existing[m.group(1)] = m.group(2)

    mem   = args.mem   or existing.get("MEM",   "320gb")
    ncpus = args.ncpus or existing.get("NCPUS", "64")
    time  = args.time  or existing.get("TIME",  "48:00:00")
    queue = args.queue or existing.get("QUEUE", "normalsr")

    redo_param = run_dir / "redo_param.py"
    lines = [
        f'export MEM="{mem}"\n',
        f'export NCPUS="{ncpus}"\n',
        f'export TIME="{time}"\n',
        f'export QUEUE="{queue}"\n',
        f'export JOB_NAME="{job_name}"\n',
    ]
    redo_param.write_text("".join(lines))
    print(f"    wrote redo_param.py  (mem={mem}, ncpus={ncpus}, time={time}, queue={queue}, job={job_name})")


def submit_job(run_dir: Path, args: argparse.Namespace) -> None:
    lz4_files = sorted(
        f for d in (run_dir / 'output').iterdir() if d.is_dir()
        for f in d.iterdir() if re.match(r'data_\d{4}\.lz4', f.name)
    )
    if not lz4_files:
        print(f"  [skip] {run_dir.name}: no checkpoint lz4 found (unexpected)")
        return

    yr       = int(lz4_files[-1].stem.split("_")[1])
    out_subs = sorted([d for d in (run_dir / "output").iterdir() if d.is_dir()], key=lambda d: d.name)

    print(f"  {run_dir.name}:")
    print(f"    checkpoint year : {yr}  ({lz4_files[-1].name})")
    print(f"    oldest output   : {out_subs[0].name if out_subs else 'none'}")

    if args.dry_run:
        print(f"    [dry-run] skipping qsub")
        return

    write_redo_param(run_dir, job_name=f"redo_{run_dir.name}", args=args)

    result = subprocess.run(
        ["bash", "redo_cmd.sh"],
        cwd=run_dir,
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"    [submitted] {result.stdout.strip()}")
    else:
        print(f"    [error] {result.stderr.strip()}")


def main():
    parser = argparse.ArgumentParser(description="Re-submit checkpoint runs on Gadi.")
    parser.add_argument("--dry-run", action="store_true",   help="Write redo_param.py but skip qsub")
    parser.add_argument("--mem",   default=None, help="PBS memory (default: inherit from task_param.py)")
    parser.add_argument("--ncpus", default=None, help="PBS CPU count (default: inherit from task_param.py)")
    parser.add_argument("--time",  default=None, help="PBS walltime (default: inherit from task_param.py)")
    parser.add_argument("--queue", default=None, help="PBS queue (default: inherit from task_param.py)")
    args = parser.parse_args()

    root = Path(__file__).parent

    print(f"Root: {root}")

    print("Querying running PBS jobs …")
    running_dirs = get_running_workdirs()

    runs = classify_runs(root, running_dirs)

    print(f"\nFinished  ({len(runs['finished'])}): {[d.name for d in runs['finished']]}")
    print(f"Running   ({len(runs['running'])}):  {[d.name for d in runs['running']]}")
    print(f"Checkpoint({len(runs['checkpoint'])}): {[d.name for d in runs['checkpoint']]}")
    print(f"Incomplete({len(runs['incomplete'])}): {[d.name for d in runs['incomplete']]}")

    if not runs["checkpoint"]:
        print("\nNo checkpoint runs to resubmit.")
        return

    print(f"\nSubmitting {len(runs['checkpoint'])} checkpoint job(s) …\n")
    for run_dir in runs["checkpoint"]:
        submit_job(run_dir, args)


if __name__ == "__main__":
    main()
