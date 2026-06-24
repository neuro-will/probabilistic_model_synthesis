"""Run one manifest row for the Figure 5 paired cross-fold sweep."""

import argparse
import json
import os
import pathlib
import subprocess


def _run_command(cmd: list):
    print("Running:", " ".join(str(v) for v in cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Worker for one Figure 5 paired cross-fold task.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest JSON.")
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="1-based task index. If omitted, uses $LSB_JOBINDEX for LSF arrays.",
    )
    args = parser.parse_args()

    if args.index is None:
        lsf_idx = os.environ.get("LSB_JOBINDEX", None)
        if lsf_idx is None:
            raise ValueError("No --index provided and LSB_JOBINDEX is not set.")
        index_1based = int(lsf_idx)
    else:
        index_1based = int(args.index)

    manifest_path = pathlib.Path(args.manifest)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    mode = manifest["mode"]
    param_file = manifest["param_file"]
    save_file = manifest["save_file"]
    test_periods = manifest["test_periods"]
    early_stopping_scope = manifest.get("early_stopping_scope", None)
    if early_stopping_scope not in {"target_only", "all_subjects"}:
        raise ValueError(
            "Manifest must set early_stopping_scope to 'target_only' or 'all_subjects'. "
            f"Got: {early_stopping_scope}"
        )
    jobs = manifest["jobs"]

    idx = index_1based - 1
    if idx < 0 or idx >= len(jobs):
        raise IndexError(
            f"Requested index {index_1based} but manifest has {len(jobs)} jobs."
        )

    spec = jobs[idx]
    type_dir = pathlib.Path(spec["type_dir"])
    fold_key = spec["fold_key"]
    fold_str_file = spec["fold_str_file"]
    rand_seed = int(spec["rand_seed"])
    tgt_subj = int(spec["tgt_subj"])

    script_dir = pathlib.Path(__file__).resolve().parent
    fit_script = script_dir / "syn_ahrens_gnldr_mdls.py"
    pp_script = script_dir / "post_process.py"

    sp_cp_dir = type_dir / "sp_cp"
    ip_cp_dir = type_dir / "ip_cp"
    results_file = type_dir / save_file
    pp_save_file = type_dir / ("pp_" + pathlib.Path(save_file).stem + ".pt")

    print("========================================")
    print(f"Task index: {index_1based}")
    print(f"Mode: {mode}")
    print(f"Target directory: {type_dir}")
    print(f"Target subject: {tgt_subj}")
    print(f"Fold key: {fold_key}")
    print(f"Fold structure file: {fold_str_file}")
    print(f"Early stopping scope: {early_stopping_scope}")
    print("========================================")

    do_fit = mode in {"fit_only", "fit_and_post"}
    do_post = mode in {"post_only", "fit_and_post"}

    if do_fit:
        fit_cmd = [
            "python",
            str(fit_script),
            str(param_file),
            "-results_dir",
            str(type_dir),
            "-fold_str_file",
            str(fold_str_file),
            "-fold",
            str(fold_key),
            "-sp_cp_dir",
            str(sp_cp_dir),
            "-ip_cp_dir",
            str(ip_cp_dir),
            "-save_file",
            str(save_file),
            "-rand_seed",
            str(rand_seed),
        ]
        _run_command(fit_cmd)

    if do_post:
        if not results_file.exists():
            raise FileNotFoundError(f"Missing fit results file for post-processing: {results_file}")

        pp_cmd = [
            "python",
            str(pp_script),
            str(results_file),
            str(pp_save_file),
            "-early_stopping",
            "True",
            "-test_periods",
            ",".join(test_periods),
            "-rand_seed",
            str(rand_seed),
        ]

        if early_stopping_scope == "target_only":
            pp_cmd += ["-early_stopping_subjects", str(tgt_subj)]
        elif early_stopping_scope != "all_subjects":
            raise ValueError(
                f"Unknown early_stopping_scope in manifest: {early_stopping_scope}"
            )

        _run_command(pp_cmd)

    print("Task complete.")


if __name__ == "__main__":
    main()
