"""
redo_checkpoint.py  —  Re-submit PBS jobs for runs that stalled mid-simulation.

A run is classified as:
  • finished   — contains Run_Archive.zip
  • running    — its directory is the PBS_O_WORKDIR of a live Gadi job
  • checkpoint — has a data_<year>.lz4 inside its output subdir
  • incomplete — none of the above

For each checkpoint run the script reads PBS settings from the run's
task_param.py, writes redo_param.py (overriding any CLI-supplied values),
then calls `bash redo_cmd.sh` from that run dir. python_script.py detects
the existing checkpoint, skips load_data(), and resumes from the saved year.

Arguments:
    --dry-run        Classify runs and print what would happen; no files written, no qsub.
    --mem   <val>    Override PBS memory    (default: inherit from task_param.py, fallback 320gb)
    --ncpus <val>    Override PBS CPU count (default: inherit from task_param.py, fallback 64)
    --time  <val>    Override PBS walltime  (default: inherit from task_param.py, fallback 48:00:00)
    --queue <val>    Override PBS queue     (default: inherit from task_param.py, fallback normalsr)

Usage:
    # Run from the task root directory (where Run_G* subdirs live)
    cd /g/data/jk53/jinzhu/LUTO/Custom_runs/REM_RES1

    python redo_checkpoint.py                        # use each run's own task_param.py settings
    python redo_checkpoint.py --dry-run              # preview only — no files written, no qsub
    python redo_checkpoint.py --time 24:00:00        # shorten walltime; other params from task_param.py
    python redo_checkpoint.py --mem 160gb --ncpus 32 # override memory and CPUs for all redo jobs
"""

import argparse
import os
import re
import subprocess
from pathlib import Path


def get_running_workdirs() -> set[Path]:
    """Return PBS_O_WORKDIR paths for all live jobs owned by $USER.

    On Gadi, `qstat -f -u USER` outputs only the short table (no attributes).
    We therefore get job IDs first, then query each with `qstat -f <id>`.
    qstat -f also wraps long lines with a leading tab; those are joined before parsing.
    """
    user = os.environ.get("USER", "")
    id_result = subprocess.run(
        ["qstat", "-u", user],
        capture_output=True, text=True,
    )
    job_ids = [
        line.split()[0]
        for line in id_result.stdout.splitlines()
        if line and "." in line.split()[0] and not line.startswith("-") and not line.startswith(" ")
    ]

    dirs: set[Path] = set()
    for job_id in job_ids:
        detail = subprocess.run(
            ["qstat", "-f", job_id],
            capture_output=True, text=True,
        )
        joined_lines: list[str] = []
        for raw in detail.stdout.splitlines():
            if raw.startswith("\t") and joined_lines:
                joined_lines[-1] += raw.strip()
            else:
                joined_lines.append(raw.strip())
        for line in joined_lines:
            if "PBS_O_WORKDIR=" not in line:
                continue
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
