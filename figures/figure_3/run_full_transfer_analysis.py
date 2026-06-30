"""Run the full Figure 3 same-condition transfer analysis sweep.

The default backend is local execution from the active Python environment. Use
``--backend lsf`` to submit the same jobs to an LSF cluster.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import pathlib
import shlex
import subprocess
import sys
import time
from typing import Dict
from typing import List


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

SAVE_FILE = "fit_results.pt"
BASE_SUBJECTS = [1, 2, 5, 6]
TGT_SUBJECTS = [8, 10, 11]
FOLD_STR_FILES = [
    "fold_str_base_14_tgt_1.json",
    "fold_str_base_14_tgt_2.json",
    "fold_str_base_14_tgt_4.json",
    "fold_str_base_14_tgt_8.json",
    "fold_str_base_14_tgt_14.json",
]
N_FOLDS = 3
TYPES = ["comb", "ind"]

PARAM_FILE = REPO_ROOT / "results/figure_3/runs/fit_params.pkl"
RESULTS_DIR = REPO_ROOT / "results/figure_3/runs"
FIT_SCRIPT = REPO_ROOT / "figures/figure_3/syn_ahrens_gnlr_mdls.py"
POST_PROCESS_SCRIPT = REPO_ROOT / "figures/figure_3/post_process.py"


def _build_jobs() -> List[Dict]:
    jobs = []
    rand_seed = 0
    for fold_str_file in FOLD_STR_FILES:
        fold_str_dir = RESULTS_DIR / pathlib.Path(fold_str_file).stem
        for fold in range(N_FOLDS):
            fold_dir = fold_str_dir / f"fold_{fold}"
            for tgt_subj in TGT_SUBJECTS:
                tgt_subj_dir = fold_dir / f"subj_{tgt_subj}"
                for fit_type in TYPES:
                    rand_seed += 1
                    type_dir = tgt_subj_dir / fit_type
                    if fit_type == "comb":
                        fit_subjects = ",".join(str(s) for s in BASE_SUBJECTS + [tgt_subj])
                    else:
                        fit_subjects = str(tgt_subj)
                    jobs.append(
                        {
                            "fold_str_file": fold_str_file,
                            "fold": fold,
                            "tgt_subj": tgt_subj,
                            "fit_type": fit_type,
                            "fit_subjects": fit_subjects,
                            "rand_seed": rand_seed,
                            "type_dir": type_dir,
                        }
                    )
    return jobs


def _fit_command(job: Dict) -> List[str]:
    type_dir = pathlib.Path(job["type_dir"])
    return [
        sys.executable,
        str(FIT_SCRIPT),
        str(PARAM_FILE),
        "-results_dir",
        str(type_dir),
        "-fold_str_file",
        job["fold_str_file"],
        "-fold",
        str(job["fold"]),
        "-sp_cp_dir",
        str(type_dir / "sp_cp"),
        "-ip_cp_dir",
        str(type_dir / "ip_cp"),
        "-subject_filter",
        job["fit_subjects"],
        "-save_file",
        SAVE_FILE,
        "-rand_seed",
        str(job["rand_seed"]),
    ]


def _post_process_command(job: Dict) -> List[str]:
    type_dir = pathlib.Path(job["type_dir"])
    results_file = type_dir / SAVE_FILE
    save_file = type_dir / f"pp_{pathlib.Path(SAVE_FILE).stem}.pkl"
    return [
        sys.executable,
        str(POST_PROCESS_SCRIPT),
        str(results_file),
        str(save_file),
        "-early_stopping_subjects",
        job["fit_subjects"],
        "-early_stopping",
        "True",
        "-rand_seed",
        str(job["rand_seed"]),
    ]


def _run_command(cmd: List[str], dry_run: bool):
    print("Running:", " ".join(shlex.quote(str(v)) for v in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def _run_local_job(job: Dict, fit: bool, post_process: bool, dry_run: bool):
    type_dir = pathlib.Path(job["type_dir"])
    if not dry_run:
        type_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Task: fold_file={job['fold_str_file']} fold={job['fold']} "
        f"target={job['tgt_subj']} fit_type={job['fit_type']}",
        flush=True,
    )
    if fit:
        _run_command(_fit_command(job), dry_run=dry_run)
    if post_process:
        _run_command(_post_process_command(job), dry_run=dry_run)


def _shell_join(cmd: List[str]) -> str:
    return " ".join(shlex.quote(str(v)) for v in cmd)


def _submit_lsf_job(job: Dict, fit: bool, post_process: bool, queue: str, conda_env: str, dry_run: bool):
    type_dir = pathlib.Path(job["type_dir"])
    if not dry_run:
        type_dir.mkdir(parents=True, exist_ok=True)
    commands = []
    if fit:
        commands.append(_shell_join(_fit_command(job)))
    if post_process:
        commands.append(_shell_join(_post_process_command(job)))
    if not commands:
        raise ValueError("At least one of fit or post_process must be enabled.")

    shell_command = " && ".join(commands)
    if conda_env:
        shell_command = (
            f"source $(conda info --base)/etc/profile.d/conda.sh && "
            f"conda activate {shlex.quote(conda_env)} && {shell_command}"
        )
    bsub_cmd = [
        "bsub",
        "-n",
        "1",
        "-gpu",
        "num=1",
        "-q",
        queue,
        "-o",
        str(type_dir / "log.txt"),
        "bash",
        "-lc",
        shell_command,
    ]
    print("Submitting:", _shell_join(bsub_cmd), flush=True)
    if not dry_run:
        subprocess.run(bsub_cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["local", "lsf"], default="local")
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel local workers.")
    parser.add_argument("--queue", default="gpu_a100", help="LSF queue for --backend lsf.")
    parser.add_argument("--conda-env", default="dpms", help="Conda env to activate for LSF jobs.")
    parser.add_argument("--submit-delay-s", type=float, default=0.0, help="Delay between LSF submissions.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--skip-fit", action="store_true")
    parser.add_argument("--skip-post-process", action="store_true")
    args = parser.parse_args()

    fit = not args.skip_fit
    post_process = not args.skip_post_process
    if not fit and not post_process:
        raise ValueError("At least one of fitting or post-processing must be enabled.")

    jobs = _build_jobs()
    print(f"Prepared {len(jobs)} Figure 3 tasks.")
    print(f"Backend: {args.backend}")

    if args.backend == "local":
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(_run_local_job, job, fit, post_process, args.dry_run)
                for job in jobs
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    else:
        for job in jobs:
            _submit_lsf_job(
                job=job,
                fit=fit,
                post_process=post_process,
                queue=args.queue,
                conda_env=args.conda_env,
                dry_run=args.dry_run,
            )
            if args.submit_delay_s > 0:
                time.sleep(args.submit_delay_s)


if __name__ == "__main__":
    main()
