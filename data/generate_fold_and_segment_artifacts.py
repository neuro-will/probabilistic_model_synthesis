"""Generate JSON fold and segment artifacts used by the manuscript scripts.

This script replaces the original notebook-only pickle artifact workflow. It
expects the manuscript subject folders under ``data/`` and writes JSON metadata
under ``data/fold_and_segment_structures/``.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

from probabilistic_model_synthesis.annotations import label_periods
from probabilistic_model_synthesis.annotations import label_subperiods
from probabilistic_model_synthesis.data_utils import load_processed_data
from probabilistic_model_synthesis.data_utils import save_json_artifact
from probabilistic_model_synthesis.data_utils import segment_dataset
from probabilistic_model_synthesis.data_utils import segment_dataset_with_constant_segment_sizes


SAME_COND_SEGMENT_FILE = "phototaxis_ns_subjects_1_2_5_6_8_9_10_11.json"
ACROSS_COND_SEGMENT_FILE = "omr_l_r_f_ns_across_cond_segments_8_9_10_11.json"
SAME_COND_FOLD_TARGET_TRAIN_COUNTS = [1, 2, 4, 8, 14]


def _portable_path(path: Path) -> str:
    """Store repository-relative metadata paths when possible."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _value_function(name: str):
    if name == "mean":
        return lambda x: np.mean(x)
    if name == "max":
        return lambda x: np.max(x)
    raise ValueError(f"Unknown value function: {name}")


def _save_segment_artifact(ps: dict, segment_tables: OrderedDict, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_tables = OrderedDict(
        (subject, table.to_dict()) for subject, table in segment_tables.items()
    )
    save_json_artifact({"ps": ps, "segment_tables": serializable_tables}, save_path)
    print(f"Saved segment tables to: {save_path}")


def generate_same_condition_segments(data_dir: Path, artifact_dir: Path):
    ps = {
        "data_dir": _portable_path(data_dir),
        "subjects": [1, 2, 5, 6, 8, 9, 10, 11],
        "n_sets": 42,
        "chunk_size": 5,
        "value_chs": [3, 4],
        "groups": OrderedDict(
            [("phototaxis_ns", [{"period": "phototaxis", "shock": False}])]
        ),
        "value_fnc": "max",
        "random_vl_assignment": True,
        "segment_labels": [f"set_{i}" for i in range(42)],
        "segment_ratios": [1] * 42,
    }

    value_fnc = _value_function(ps["value_fnc"])
    segment_tables = OrderedDict()
    for subject_id in ps["subjects"]:
        dataset = load_processed_data(data_dir / f"subject_{subject_id}")
        labels = label_periods(dataset.ts_data["stim"]["vls"][:])
        values = dataset.ts_data["behavior"]["vls"][:, ps["value_chs"]]
        subj_values = np.mean(values, axis=1)
        segment_tables[subject_id] = segment_dataset(
            period_lbls=labels,
            groups=ps["groups"],
            chunk_size=ps["chunk_size"],
            segment_labels=ps["segment_labels"],
            segment_ratios=ps["segment_ratios"],
            vls=subj_values,
            vl_fnc=value_fnc,
            random_vl_assignment=ps["random_vl_assignment"],
        )
        print(f"Generated same-condition segments for subject {subject_id}.")

    _save_segment_artifact(ps, segment_tables, artifact_dir / SAME_COND_SEGMENT_FILE)


def generate_across_condition_segments(data_dir: Path, artifact_dir: Path):
    ps = {
        "data_dir": _portable_path(data_dir),
        "subjects": [8, 9, 10, 11],
        "chunk_size": 5,
        "value_chs": [3, 4],
        "groups": OrderedDict(
            [
                ("omr_l_ns", [{"period": "omr_left", "shock": False}]),
                ("omr_r_ns", [{"period": "omr_right", "shock": False}]),
                ("omr_f_ns", [{"period": "omr_forward", "shock": False}]),
            ]
        ),
        "value_fnc": "max",
        "n_segment_chunks": 1,
        "random_vl_assignment": False,
    }

    value_fnc = _value_function(ps["value_fnc"])
    segment_tables = OrderedDict()
    for subject_id in ps["subjects"]:
        dataset = load_processed_data(data_dir / f"subject_{subject_id}")
        labels = label_subperiods(dataset.ts_data["stim"]["vls"][:])
        values = dataset.ts_data["behavior"]["vls"][:, ps["value_chs"]]
        subj_values = np.mean(values, axis=1)
        segment_tables[subject_id] = segment_dataset_with_constant_segment_sizes(
            period_lbls=labels,
            groups=ps["groups"],
            chunk_size=ps["chunk_size"],
            n_segment_chunks=ps["n_segment_chunks"],
            vls=subj_values,
            vl_fnc=value_fnc,
            random_vl_assignment=ps["random_vl_assignment"],
        )
        print(f"Generated across-condition segments for subject {subject_id}.")

    _save_segment_artifact(ps, segment_tables, artifact_dir / ACROSS_COND_SEGMENT_FILE)


def _rotate(values: list, n: int) -> list:
    return values[n:] + values[:n]


def generate_same_condition_folds(artifact_dir: Path):
    n_sets = 42
    n_folds = 3
    base_subjects = [1, 2, 5, 6]
    target_subjects = [8, 9, 10, 11]
    subjects = base_subjects + target_subjects
    groups = ["phototaxis_ns"]

    if n_sets % n_folds != 0:
        raise ValueError("Number of folds does not evenly divide the number of segment sets.")

    n_test_sets = int(n_sets / n_folds)
    test_sets = [
        list(range(i, i + n_test_sets))
        for i in np.linspace(0, n_sets, n_folds + 1, dtype=int)[:-1]
    ]
    possible_train_sets = _rotate(test_sets, 1)
    possible_validation_sets = _rotate(test_sets, 2)

    for target_train_count in SAME_COND_FOLD_TARGET_TRAIN_COUNTS:
        n_train_sets = {subject: 14 for subject in base_subjects}
        n_train_sets.update({subject: target_train_count for subject in target_subjects})
        n_validation_sets = dict(n_train_sets)

        fold_groups = {}
        for subject in subjects:
            subject_fold_groups = {}
            for fold_i in range(n_folds):
                fold_test_sets = [f"set_{s}" for s in test_sets[fold_i]]
                fold_train_sets = [
                    f"set_{s}"
                    for s in possible_train_sets[fold_i][: n_train_sets[subject]]
                ]
                fold_validation_sets = [
                    f"set_{s}"
                    for s in possible_validation_sets[fold_i][: n_validation_sets[subject]]
                ]
                subject_fold_groups[fold_i] = {
                    "test": {group: fold_test_sets for group in groups},
                    "train": {group: fold_train_sets for group in groups},
                    "validation": {group: fold_validation_sets for group in groups},
                }
            fold_groups[subject] = subject_fold_groups

        save_path = artifact_dir / f"fold_str_base_14_tgt_{target_train_count}.json"
        save_json_artifact(fold_groups, save_path)
        print(f"Saved same-condition fold structure to: {save_path}")


def generate_figure5_folds(artifact_dir: Path):
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "figures/figure_5/build_across_cond_disjoint_folds.py"),
            "--segment_table_path",
            str(artifact_dir / ACROSS_COND_SEGMENT_FILE),
            "--save_dir",
            str(artifact_dir / "gnldr_paired"),
            "--n_folds",
            "6",
            "--min_train_segments",
            "18",
            "--min_validation_segments",
            "4",
            "--min_test_segments",
            "5",
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=REPO_ROOT / "data/fold_and_segment_structures",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.random_seed)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    generate_same_condition_segments(args.data_dir, args.artifact_dir)
    generate_across_condition_segments(args.data_dir, args.artifact_dir)
    generate_same_condition_folds(args.artifact_dir)
    generate_figure5_folds(args.artifact_dir)
    print("Fold and segment artifact generation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
