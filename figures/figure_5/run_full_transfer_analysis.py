"""Run the full Figure 5 paired cross-fold transfer analysis sweep.

The default backend is local execution from the active Python environment. Use
``--backend lsf`` to submit the same manifest as an LSF array job.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
from typing import Dict
from typing import List


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

FIT = True
POST_PROCESS = True

SAVE_FILE = "fit_results.pt"
TGT_SUBJECTS = [8, 9, 11]
TRAIN_CONDITIONS = ["omr_f_ns", "omr_r_ns", "omr_l_ns"]
N_DISJOINT_FOLDS = 6
TYPES = ["multi_cond", "single_cond"]
TEST_PERIODS = ["omr_forward", "omr_right", "omr_left"]
EARLY_STOPPING_SCOPE = "target_only"

FOLD_STR_PRE_STR = "ac_an_disjoint_paired"
FOLD_STR_APP_STR = "folds.json"

PARAM_FILE = REPO_ROOT / "results/publication_results/gnldr/quantification_paired/fit_params_paired.pkl"
RESULTS_DIR = REPO_ROOT / "results/publication_results/gnldr/quantification_paired"

N_SLOTS = 1
N_GPU = 1
JOB_NAME = "fig5_paired"


def _fold_key(train_condition: str, fold_idx: int) -> str:
    return f"{train_condition}__fold_{fold_idx}"


def _fold_file_name(tgt_subj: int, fit_type: str) -> str:
    return f"{FOLD_STR_PRE_STR}_k{N_DISJOINT_FOLDS}_tgt_{tgt_subj}_{fit_type}_{FOLD_STR_APP_STR}"


def _mode_str(fit: bool, post_process: bool) -> str:
    if fit and post_process:
        return "fit_and_post"
    if fit and not post_process:
        return "fit_only"
    if (not fit) and post_process:
        return "post_only"
    raise ValueError("At least one of fitting or post-processing must be enabled.")


def _validate_early_stopping_scope(scope: str):
    if scope not in {"target_only", "all_subjects"}:
        raise ValueError(
            f"EARLY_STOPPING_SCOPE must be 'target_only' or 'all_subjects', got: {scope}"
        )


def _build_jobs() -> List[Dict]:
    job_specs = []
    rand_seed = 0
    for train_condition in TRAIN_CONDITIONS:
        for fold_idx in range(N_DISJOINT_FOLDS):
            fold_key = _fold_key(train_condition=train_condition, fold_idx=fold_idx)
            cond_fold_dir = RESULTS_DIR / train_condition / f"fold_{fold_idx}"
            for tgt_subj in TGT_SUBJECTS:
                for fit_type in TYPES:
                    rand_seed += 1
                    type_dir = cond_fold_dir / f"subj_{tgt_subj}" / fit_type
                    job_specs.append(
                        {
                            "train_condition": train_condition,
                            "fold_idx": fold_idx,
                            "fold_key": fold_key,
                            "tgt_subj": tgt_subj,
                            "fit_type": fit_type,
                            "rand_seed": rand_seed,
                            "type_dir": str(type_dir),
                            "fold_str_file": _fold_file_name(tgt_subj=tgt_subj, fit_type=fit_type),
                        }
                    )
    return job_specs


def _write_manifest(mode: str, jobs: List[Dict], dry_run: bool) -> pathlib.Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if dry_run:
        manifest_dir = pathlib.Path(tempfile.gettempdir()) / "dpms_figure_5_manifests"
    else:
        manifest_dir = RESULTS_DIR / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"manifest_{timestamp}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": mode,
                "param_file": str(PARAM_FILE),
                "save_file": SAVE_FILE,
                "test_periods": TEST_PERIODS,
                "early_stopping_scope": EARLY_STOPPING_SCOPE,
                "fold_family": FOLD_STR_PRE_STR,
                "target_fish_splits_identical_for_db_sb": True,
                "target_test_groups_disjoint_across_folds": True,
                "transfer_fish_train_only": True,
                "jobs": jobs,
            },
            f,
            indent=2,
        )
    return manifest_path


def _parse_job_id(bsub_output: str) -> str:
    match = re.search(r"Job <(\d+)>", bsub_output)
    if match is None:
        raise RuntimeError(f"Could not parse LSF job id from bsub output:\n{bsub_output}")
    return match.group(1)


def _shell_join(cmd: List[str]) -> str:
    return " ".join(shlex.quote(str(v)) for v in cmd)


def _worker_command(manifest_path: pathlib.Path, index: int) -> List[str]:
    worker_script = pathlib.Path(__file__).resolve().parent / "sweep_worker.py"
    return [
        sys.executable,
        str(worker_script),
        "--manifest",
        str(manifest_path),
        "--index",
        str(index),
    ]


def _run_local_task(manifest_path: pathlib.Path, index: int, dry_run: bool):
    cmd = _worker_command(manifest_path, index)
    print("Running:", _shell_join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def _submit_lsf_array(
    manifest_path: pathlib.Path,
    n_jobs: int,
    max_parallel: int,
    queue: str,
    conda_env: str,
    dry_run: bool,
):
    log_dir = RESULTS_DIR / "_array_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    worker_script = pathlib.Path(__file__).resolve().parent / "sweep_worker.py"
    array_name = f"{JOB_NAME}[1-{n_jobs}]%{max_parallel}"
    shell_command = f"{sys.executable} {shlex.quote(str(worker_script))} --manifest {shlex.quote(str(manifest_path))}"
    if conda_env:
        shell_command = (
            f"source $(conda info --base)/etc/profile.d/conda.sh && "
            f"conda activate {shlex.quote(conda_env)} && {shell_command}"
        )

    bsub_cmd = [
        "bsub",
        "-n",
        str(N_SLOTS),
        "-gpu",
        f"num={N_GPU}",
        "-q",
        queue,
        "-J",
        array_name,
        "-oo",
        str(log_dir / f"{JOB_NAME}_%J_%I.out"),
        "-eo",
        str(log_dir / f"{JOB_NAME}_%J_%I.err"),
        "bash",
        "-lc",
        shell_command,
    ]

    print("Submitting:", _shell_join(bsub_cmd), flush=True)
    if dry_run:
        return
    completed = subprocess.run(bsub_cmd, check=True, capture_output=True, text=True)
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip())
    print(f"Submitted array job id: {_parse_job_id(completed.stdout)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["local", "lsf"], default="local")
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel local workers.")
    parser.add_argument("--queue", default="gpu_a100", help="LSF queue for --backend lsf.")
    parser.add_argument("--max-parallel-array-jobs", type=int, default=50)
    parser.add_argument("--conda-env", default="dpms", help="Conda env to activate for LSF jobs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--skip-fit", action="store_true")
    parser.add_argument("--skip-post-process", action="store_true")
    args = parser.parse_args()

    _validate_early_stopping_scope(EARLY_STOPPING_SCOPE)
    mode = _mode_str(fit=not args.skip_fit, post_process=not args.skip_post_process)
    jobs = _build_jobs()
    manifest_path = _write_manifest(mode=mode, jobs=jobs, dry_run=args.dry_run)

    print(f"Prepared {len(jobs)} Figure 5 tasks.")
    print(f"Backend: {args.backend}")
    print(f"Mode: {mode}")
    print(f"Early stopping scope: {EARLY_STOPPING_SCOPE}")
    print(f"Manifest: {manifest_path}")

    if args.backend == "local":
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(_run_local_task, manifest_path, index, args.dry_run)
                for index in range(1, len(jobs) + 1)
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    else:
        _submit_lsf_array(
            manifest_path=manifest_path,
            n_jobs=len(jobs),
            max_parallel=args.max_parallel_array_jobs,
            queue=args.queue,
            conda_env=args.conda_env,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
