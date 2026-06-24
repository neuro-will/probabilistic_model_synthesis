"""Build paired fold structures for across-condition transfer analysis.

This script generates fold structures for both:
1. multi_cond (DB): transfer fish are trained on non-target conditions
2. single_cond (SB): all fish are trained on the target condition

For each target fish and target condition, folds are keyed as:
    <target_condition>__fold_<k>

Each fold keeps the target fish exactly paired between DB and SB: identical
target-fish train, validation, and test segment IDs. Transfer fish are train-only,
matching the original Figure 5 design. Target-fish test segments are disjoint
across folds for every tested condition.
"""

import argparse
import pathlib
from typing import Dict
from typing import Optional
from typing import List
from typing import Sequence
from typing import Set
from typing import Tuple

import numpy as np

from probabilistic_model_synthesis.data_utils import SegmentTable
from probabilistic_model_synthesis.data_utils import load_segment_tables
from probabilistic_model_synthesis.data_utils import save_json_artifact


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _portable_path(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _to_set_labels(seg_ids: Sequence[int]) -> List[str]:
    return [f"set_{int(i)}" for i in seg_ids]


def _to_id_set(seg_labels: Sequence[str]) -> Set[int]:
    return {int(lbl.split("_")[1]) for lbl in seg_labels}


def _partition_ids(n_segments: int, n_folds: int, rng: np.random.Generator) -> List[np.ndarray]:
    ids = np.arange(n_segments, dtype=int)
    rng.shuffle(ids)
    return [np.sort(chunk) for chunk in np.array_split(ids, n_folds)]


def _partition_equal_size_ids(
    n_segments: int,
    n_folds: int,
    n_per_fold: int,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    n_required = n_folds * n_per_fold
    if n_required > n_segments:
        raise ValueError(
            f"Requested {n_required} test segments from only {n_segments} available segments."
        )
    ids = np.arange(n_segments, dtype=int)
    rng.shuffle(ids)
    ids = ids[:n_required]
    return [np.sort(chunk) for chunk in np.split(ids, n_folds)]


def _sample_ids(
    n_segments: int,
    n_requested: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_requested > n_segments:
        raise ValueError(f"Requested {n_requested} segments from only {n_segments} available segments.")
    ids = np.arange(n_segments, dtype=int)
    rng.shuffle(ids)
    return np.sort(ids[:n_requested])


def _split_train_val_from_remaining(
    n_segments: int,
    test_ids: np.ndarray,
    n_train_segments: int,
    n_validation_segments: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    remaining = np.setdiff1d(np.arange(n_segments, dtype=int), test_ids, assume_unique=True)
    if (n_train_segments + n_validation_segments) > remaining.size:
        raise ValueError(
            f"Requested train+validation segments ({n_train_segments + n_validation_segments}) "
            f"exceed available non-test segments ({remaining.size})."
        )
    remaining = remaining.copy()
    rng.shuffle(remaining)
    val_ids = np.sort(remaining[:n_validation_segments])
    train_ids = np.sort(remaining[n_validation_segments:n_validation_segments + n_train_segments])
    return train_ids, val_ids


def _compute_train_val_sizes(
    n_segments_per_pair: Dict[Tuple[int, str], int],
    test_bins_per_pair: Dict[Tuple[int, str], List[np.ndarray]],
    n_train_segments: int,
    n_validation_segments: int,
    train_fraction: float,
    validation_fraction: float,
    min_train_segments: int,
    min_validation_segments: int,
) -> Tuple[int, int]:
    min_total_segments = min(n_segments_per_pair.values())
    min_available_after_test = min(
        n_segments_per_pair[pair] - max(len(bin_ids) for bin_ids in test_bins_per_pair.get(pair, [np.array([])]))
        for pair in n_segments_per_pair.keys()
    )
    if min_available_after_test <= 0:
        raise ValueError("No non-test segments remain for at least one required fish-condition pair.")

    if n_validation_segments is None:
        n_val = max(min_validation_segments, int(np.floor(validation_fraction * min_total_segments)))
    else:
        n_val = int(n_validation_segments)

    if n_train_segments is None:
        n_train = max(min_train_segments, int(np.floor(train_fraction * min_total_segments)))
    else:
        n_train = int(n_train_segments)

    if n_val < min_validation_segments:
        raise ValueError(
            f"Configured validation segments ({n_val}) is below min_validation_segments ({min_validation_segments})."
        )
    if n_train < min_train_segments:
        raise ValueError(f"Configured train segments ({n_train}) is below min_train_segments ({min_train_segments}).")

    if (n_train + n_val) > min_available_after_test:
        # Keep validation fixed first and trim train to the feasible max.
        n_train = min_available_after_test - n_val
        if n_train < min_train_segments:
            raise ValueError(
                "Not enough non-test segments to satisfy min train/validation constraints. "
                f"Feasible non-test min: {min_available_after_test}, "
                f"requested min train: {min_train_segments}, min validation: {min_validation_segments}."
            )

    return n_train, n_val


def _build_target_entry(
    target_subject: int,
    target_group: str,
    groups: Sequence[str],
    seg_tables: Dict[int, SegmentTable],
    fold_idx: int,
    target_test_bins: Dict[str, List[np.ndarray]],
    n_train_segments: int,
    n_validation_segments: int,
    rng: np.random.Generator,
) -> Dict[str, Dict[str, List[str]]]:
    n_train_group_segments = seg_tables[target_subject].n_group_segments(target_group)
    target_train_ids, target_val_ids = _split_train_val_from_remaining(
        n_segments=n_train_group_segments,
        test_ids=target_test_bins[target_group][fold_idx],
        n_train_segments=n_train_segments,
        n_validation_segments=n_validation_segments,
        rng=rng,
    )

    test = {}
    for group in groups:
        test[group] = _to_set_labels(target_test_bins[group][fold_idx])

    return {
        "train": {target_group: _to_set_labels(target_train_ids)},
        "validation": {target_group: _to_set_labels(target_val_ids)},
        "test": test,
    }


def _build_one_schema_fold_dict(
    subjects: Sequence[int],
    target_subject: int,
    target_entry: Dict[str, Dict[str, List[str]]],
    schema_train_group_map: Dict[int, str],
    seg_tables: Dict[int, SegmentTable],
    fold_key: str,
    n_train_segments: int,
    rng: np.random.Generator,
) -> Dict[int, Dict[str, Dict[str, List[str]]]]:
    out = {}
    for subject in subjects:
        if subject == target_subject:
            out[subject] = {fold_key: target_entry}
            continue

        train_group = schema_train_group_map[subject]
        n_segments = seg_tables[subject].n_group_segments(train_group)
        train_ids = _sample_ids(n_segments=n_segments, n_requested=n_train_segments, rng=rng)
        entry = {
            "train": {train_group: _to_set_labels(train_ids)},
            "validation": None,
            "test": None,
        }
        out[subject] = {fold_key: entry}
    return out


def _count_points_for_group_dict(
    segment_table: SegmentTable,
    group_dict: Optional[Dict[str, Sequence[str]]],
) -> Optional[int]:
    if group_dict is None:
        return None

    n_points = 0
    for group, segment_labels in group_dict.items():
        for segment_label in segment_labels:
            slices = segment_table.find(segment=segment_label, group=group)
            if slices is None:
                continue
            n_points += sum(sl.stop - sl.start for sl in slices)
    return n_points


def _sample_count_summary(
    fold_struct: Dict[int, dict],
    seg_tables: Dict[int, SegmentTable],
) -> Dict[int, Dict[str, Dict[str, Optional[int]]]]:
    summary = {}
    for subject, fold_entries in fold_struct.items():
        summary[subject] = {}
        for fold_key, entry in fold_entries.items():
            summary[subject][fold_key] = {
                cv_set: _count_points_for_group_dict(seg_tables[subject], entry[cv_set])
                for cv_set in ["train", "validation", "test"]
            }
    return summary


def _merge_subject_fold_dicts(base: Dict[int, dict], update: Dict[int, dict]) -> Dict[int, dict]:
    for subject, fold_dict in update.items():
        if subject not in base:
            base[subject] = {}
        for fold_key, fold_entry in fold_dict.items():
            if fold_key in base[subject]:
                raise ValueError(f"Duplicate fold key encountered for subject {subject}: {fold_key}")
            base[subject][fold_key] = fold_entry
    return base


def _validate_fold_structures(
    multi_cond_folds: Dict[int, dict],
    single_cond_folds: Dict[int, dict],
    seg_tables: Dict[int, SegmentTable],
    target_subject: int,
    transfer_subjects: Sequence[int],
    groups: Sequence[str],
    n_folds: int,
):
    target_train_counts = []
    target_train_sample_counts = []
    all_train_sample_counts = []
    target_validation_counts = []
    target_validation_sample_counts = []
    target_test_counts_by_group = {group: [] for group in groups}
    target_test_sample_counts_by_group = {group: [] for group in groups}
    for fold_key in multi_cond_folds[target_subject].keys():
        if multi_cond_folds[target_subject][fold_key] != single_cond_folds[target_subject][fold_key]:
            raise ValueError(f"Target fish SB/DB split mismatch for fold {fold_key}.")

        target_entry = multi_cond_folds[target_subject][fold_key]
        target_train_counts.append(
            sum(len(v) for v in target_entry["train"].values())
        )
        target_train_sample_counts.append(
            _count_points_for_group_dict(seg_tables[target_subject], target_entry["train"])
        )
        all_train_sample_counts.append(target_train_sample_counts[-1])
        target_validation_counts.append(
            sum(len(v) for v in target_entry["validation"].values())
        )
        target_validation_sample_counts.append(
            _count_points_for_group_dict(seg_tables[target_subject], target_entry["validation"])
        )
        for group in groups:
            target_test_counts_by_group[group].append(len(target_entry["test"][group]))
            target_test_sample_counts_by_group[group].append(
                _count_points_for_group_dict(seg_tables[target_subject], {group: target_entry["test"][group]})
            )

    if len(set(target_train_counts)) != 1:
        raise ValueError(f"Target train segment counts differ across folds: {target_train_counts}")
    if len(set(target_train_sample_counts)) != 1:
        raise ValueError(f"Target train sample counts differ across folds: {target_train_sample_counts}")
    if len(set(target_validation_counts)) != 1:
        raise ValueError(f"Target validation segment counts differ across folds: {target_validation_counts}")
    if len(set(target_validation_sample_counts)) != 1:
        raise ValueError(
            f"Target validation sample counts differ across folds: {target_validation_sample_counts}"
        )
    for group, counts in target_test_counts_by_group.items():
        if len(set(counts)) != 1:
            raise ValueError(f"Target test segment counts differ for group {group}: {counts}")
    for group, counts in target_test_sample_counts_by_group.items():
        if len(set(counts)) != 1:
            raise ValueError(f"Target test sample counts differ for group {group}: {counts}")

    for subject in transfer_subjects:
        transfer_train_counts = []
        transfer_train_sample_counts = []
        for fold_key, entry in multi_cond_folds[subject].items():
            if entry["validation"] is not None or entry["test"] is not None:
                raise ValueError(f"Transfer fish {subject} has validation/test data in DB fold {fold_key}.")
            transfer_train_counts.append(sum(len(v) for v in entry["train"].values()))
            transfer_train_sample_counts.append(_count_points_for_group_dict(seg_tables[subject], entry["train"]))
            all_train_sample_counts.append(transfer_train_sample_counts[-1])
        for fold_key, entry in single_cond_folds[subject].items():
            if entry["validation"] is not None or entry["test"] is not None:
                raise ValueError(f"Transfer fish {subject} has validation/test data in SB fold {fold_key}.")
            transfer_train_counts.append(sum(len(v) for v in entry["train"].values()))
            transfer_train_sample_counts.append(_count_points_for_group_dict(seg_tables[subject], entry["train"]))
            all_train_sample_counts.append(transfer_train_sample_counts[-1])
        if len(set(transfer_train_counts)) != 1:
            raise ValueError(
                f"Transfer train segment counts differ for subject {subject}: {transfer_train_counts}"
            )
        if len(set(transfer_train_sample_counts)) != 1:
            raise ValueError(
                f"Transfer train sample counts differ for subject {subject}: {transfer_train_sample_counts}"
            )

    if len(set(all_train_sample_counts)) != 1:
        raise ValueError(f"Train sample counts differ across fit entries: {all_train_sample_counts}")

    for target_group in groups:
        for test_group in groups:
            test_sets = []
            for fold_idx in range(n_folds):
                fold_key = f"{target_group}__fold_{fold_idx}"
                test_sets.append(_to_id_set(multi_cond_folds[target_subject][fold_key]["test"][test_group]))
            overlap = sum(
                len(test_sets[i] & test_sets[j])
                for i in range(len(test_sets))
                for j in range(i + 1, len(test_sets))
            )
            if overlap != 0:
                raise ValueError(
                    f"Target test folds overlap for target_group={target_group}, "
                    f"test_group={test_group}, overlap={overlap}."
                )


def main():
    parser = argparse.ArgumentParser(
        description="Generate disjoint multi-fold structures for across-condition transfer analysis."
    )
    parser.add_argument(
        "--segment_table_path",
        type=str,
        default="data/fold_and_segment_structures/omr_l_r_f_ns_across_cond_segments_8_9_10_11.json",
        help="Path to segment table JSON, or legacy pickle.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/fold_and_segment_structures/gnldr_paired",
        help="Directory to save generated fold-structure files.",
    )
    parser.add_argument(
        "--save_prefix",
        type=str,
        default="ac_an_disjoint_paired",
        help="Prefix for output fold files.",
    )
    parser.add_argument("--subjects", type=str, default="8,9,11", help="Comma-separated target subjects.")
    parser.add_argument("--groups", type=str, default="omr_f_ns,omr_r_ns,omr_l_ns", help="Comma-separated groups.")
    parser.add_argument("--n_folds", type=int, default=6, help="Number of disjoint test folds per condition.")
    parser.add_argument(
        "--n_train_segments",
        type=int,
        default=None,
        help="Exact train segments per subject-condition per fold. If not set, auto-derived.",
    )
    parser.add_argument(
        "--n_validation_segments",
        type=int,
        default=None,
        help="Exact validation segments per subject-condition per fold. If not set, auto-derived.",
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.7,
        help="Used only when n_train_segments is not provided.",
    )
    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.15,
        help="Used only when n_validation_segments is not provided.",
    )
    parser.add_argument("--min_train_segments", type=int, default=18, help="Minimum train segments.")
    parser.add_argument("--min_validation_segments", type=int, default=4, help="Minimum validation segments.")
    parser.add_argument("--min_test_segments", type=int, default=5, help="Minimum test segments in every fold.")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_seed)
    subjects = [int(v) for v in args.subjects.split(",")]
    groups = [g.strip() for g in args.groups.split(",")]

    segment_table_path = pathlib.Path(args.segment_table_path)
    seg_tables = load_segment_tables(segment_table_path)

    missing_subjects = [s for s in subjects if s not in seg_tables]
    if missing_subjects:
        raise ValueError(f"Subjects missing from segment table: {missing_subjects}")

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for tgt_subject in subjects:
        multi_cond_folds = {}
        single_cond_folds = {}
        transfer_subjects = [s for s in subjects if s != tgt_subject]

        target_test_segments_per_fold = min(
            seg_tables[tgt_subject].n_group_segments(group) // args.n_folds
            for group in groups
        )
        if target_test_segments_per_fold < args.min_test_segments:
            raise ValueError(
                f"Cannot satisfy min_test_segments={args.min_test_segments} for "
                f"target_subject={tgt_subject}. The largest equal-size disjoint "
                f"test fold has {target_test_segments_per_fold} segments."
            )

        # Reuse the same target-fish test bins for every target training condition.
        target_test_bins = {}
        for group in groups:
            n_segments = seg_tables[tgt_subject].n_group_segments(group)
            target_test_bins[group] = _partition_equal_size_ids(
                n_segments=n_segments,
                n_folds=args.n_folds,
                n_per_fold=target_test_segments_per_fold,
                rng=rng,
            )

        # Use one global train/validation size per target analysis so train
        # condition, fold, and SB/DB comparisons are all count-matched.
        all_required_pairs = {(s, group) for s in subjects for group in groups}
        n_segments_per_pair = {
            pair: seg_tables[pair[0]].n_group_segments(pair[1])
            for pair in all_required_pairs
        }
        n_train, n_val = _compute_train_val_sizes(
            n_segments_per_pair=n_segments_per_pair,
            test_bins_per_pair={
                (tgt_subject, group): target_test_bins[group]
                for group in groups
            },
            n_train_segments=args.n_train_segments,
            n_validation_segments=args.n_validation_segments,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            min_train_segments=args.min_train_segments,
            min_validation_segments=args.min_validation_segments,
        )

        for tgt_group in groups:
            transfer_groups = [g for g in groups if g != tgt_group]
            if len(transfer_subjects) != len(transfer_groups):
                raise ValueError(
                    "For this setup we expect exactly one transfer group per transfer subject. "
                    f"Got {len(transfer_subjects)} transfer subjects and {len(transfer_groups)} transfer groups."
                )

            multi_train_group_map = {tgt_subject: tgt_group}
            for s, g in zip(transfer_subjects, transfer_groups):
                multi_train_group_map[s] = g

            single_train_group_map = {s: tgt_group for s in subjects}

            for fold_idx in range(args.n_folds):
                fold_key = f"{tgt_group}__fold_{fold_idx}"
                target_entry = _build_target_entry(
                    target_subject=tgt_subject,
                    target_group=tgt_group,
                    groups=groups,
                    seg_tables=seg_tables,
                    fold_idx=fold_idx,
                    target_test_bins=target_test_bins,
                    n_train_segments=n_train,
                    n_validation_segments=n_val,
                    rng=rng,
                )

                multi_fold = _build_one_schema_fold_dict(
                    subjects=subjects,
                    target_subject=tgt_subject,
                    target_entry=target_entry,
                    schema_train_group_map=multi_train_group_map,
                    seg_tables=seg_tables,
                    fold_key=fold_key,
                    n_train_segments=n_train,
                    rng=rng,
                )
                single_fold = _build_one_schema_fold_dict(
                    subjects=subjects,
                    target_subject=tgt_subject,
                    target_entry=target_entry,
                    schema_train_group_map=single_train_group_map,
                    seg_tables=seg_tables,
                    fold_key=fold_key,
                    n_train_segments=n_train,
                    rng=rng,
                )

                multi_cond_folds = _merge_subject_fold_dicts(multi_cond_folds, multi_fold)
                single_cond_folds = _merge_subject_fold_dicts(single_cond_folds, single_fold)

        _validate_fold_structures(
            multi_cond_folds=multi_cond_folds,
            single_cond_folds=single_cond_folds,
            seg_tables=seg_tables,
            target_subject=tgt_subject,
            transfer_subjects=transfer_subjects,
            groups=groups,
            n_folds=args.n_folds,
        )

        multi_file = save_dir / f"{args.save_prefix}_k{args.n_folds}_tgt_{tgt_subject}_multi_cond_folds.json"
        single_file = save_dir / f"{args.save_prefix}_k{args.n_folds}_tgt_{tgt_subject}_single_cond_folds.json"
        save_json_artifact(multi_cond_folds, multi_file)
        save_json_artifact(single_cond_folds, single_file)

        summary = {
            "target_subject": tgt_subject,
            "groups": groups,
            "subjects": subjects,
            "n_folds": args.n_folds,
            "n_train_segments_per_fold": n_train,
            "n_validation_segments_per_fold": n_val,
            "n_test_segments_per_fold": target_test_segments_per_fold,
            "min_test_segments": args.min_test_segments,
            "target_fish_splits_identical_for_db_sb": True,
            "target_test_bins_reused_across_train_conditions": True,
            "target_test_groups_disjoint_across_folds": True,
            "train_validation_test_segment_counts_matched_across_folds": True,
            "transfer_fish_train_only": True,
            "random_seed": args.random_seed,
            "segment_table_path": _portable_path(segment_table_path),
            "multi_cond_sample_counts": _sample_count_summary(multi_cond_folds, seg_tables),
            "single_cond_sample_counts": _sample_count_summary(single_cond_folds, seg_tables),
        }
        summary_file = save_dir / f"{args.save_prefix}_k{args.n_folds}_tgt_{tgt_subject}_summary.json"
        save_json_artifact(summary, summary_file)

        print(f"Saved fold files for target subject {tgt_subject}:")
        print(f"  {multi_file}")
        print(f"  {single_file}")
        print(
            "  Split sizes per fold: "
            f"train={n_train}, validation={n_val}, "
            f"test={target_test_segments_per_fold}."
        )


if __name__ == "__main__":
    main()
