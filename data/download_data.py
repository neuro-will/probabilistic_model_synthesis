"""Download the Chen et al. zebrafish data used by the DPMS manuscript.

The Figshare/Janelia DOI contains more subjects than this manuscript uses.
By default, this script downloads only subjects used by the DPMS analyses:
1, 2, 5, 6, 8, 9, 10, and 11, plus the additional MAT archive containing
shared reference files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import zipfile
import zlib


ARTICLE_API_URL = "https://api.figshare.com/v2/articles/7272617"
DEFAULT_SUBJECTS = [1, 2, 5, 6, 8, 9, 10, 11]
DOWNLOADED_SUBJECT_FILES = ["data_full.mat", "TimeSeries.h5"]
ANALYSIS_READY_SUBJECT_FILES = DOWNLOADED_SUBJECT_FILES
REPO_ROOT = Path(__file__).resolve().parents[1]
ADDITIONAL_MAT_FILES = [
    "CustomColormaps.mat",
    "FishOutline.mat",
    "FishOutline_woEyes.mat",
    "MaskDatabase.mat",
    "ReferenceBrain.mat",
    "VAR_new.mat",
]
REQUIRED_ARTIFACT_FILES = [
    "phototaxis_ns_subjects_1_2_5_6_8_9_10_11.json",
    "omr_l_r_f_ns_across_cond_segments_8_9_10_11.json",
    "fold_str_base_14_tgt_1.json",
    "fold_str_base_14_tgt_2.json",
    "fold_str_base_14_tgt_4.json",
    "fold_str_base_14_tgt_8.json",
    "fold_str_base_14_tgt_14.json",
    "gnldr_paired/ac_an_disjoint_paired_k6_tgt_8_multi_cond_folds.json",
    "gnldr_paired/ac_an_disjoint_paired_k6_tgt_8_single_cond_folds.json",
    "gnldr_paired/ac_an_disjoint_paired_k6_tgt_9_multi_cond_folds.json",
    "gnldr_paired/ac_an_disjoint_paired_k6_tgt_9_single_cond_folds.json",
    "gnldr_paired/ac_an_disjoint_paired_k6_tgt_11_multi_cond_folds.json",
    "gnldr_paired/ac_an_disjoint_paired_k6_tgt_11_single_cond_folds.json",
]


def _subject_file_name(subject: int) -> str:
    return "subject_09.zip" if subject == 9 else f"subject_{subject}.zip"


def _subject_dir_name(subject: int) -> str:
    return f"subject_{subject}"


def _fetch_article_metadata() -> dict:
    with urllib.request.urlopen(ARTICLE_API_URL, timeout=60) as response:
        return json.load(response)


def _download_file(url: str, dest: Path, expected_size: int, chunk_size: int = 1024 * 1024):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    if tmp_dest.exists():
        tmp_dest.unlink()

    start_time = time.time()
    last_report_time = start_time
    downloaded = 0
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp_dest, "wb") as out:
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            print(f"  Server Content-Length: {_format_size(int(content_length))}")
        else:
            print("  Server Content-Length: unavailable")
        print(f"  Expected file size: {_format_size(expected_size)}")

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)

            now = time.time()
            if now - last_report_time >= 5 or downloaded == expected_size:
                elapsed = max(now - start_time, 1e-9)
                rate = downloaded / elapsed
                pct = 100 * downloaded / expected_size if expected_size else 0
                print(
                    f"  Downloaded {_format_size(downloaded)} / {_format_size(expected_size)} "
                    f"({pct:.1f}%, {_format_size(rate)}/s)"
                )
                last_report_time = now

    print(f"  Final downloaded size: {_format_size(downloaded)}")
    tmp_dest.replace(dest)


def _zip_is_valid(zip_path: Path) -> bool:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False


def _zip_can_be_listed(zip_path: Path) -> bool:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            return len(zf.infolist()) > 0
    except zipfile.BadZipFile:
        return False


def _local_zip_headers(zip_path: Path) -> list[dict]:
    """Read local file headers using the central directory as the source of truth.

    Some upstream subject archives exceed 4 GiB but appear to have been written
    without correct Zip64 local-header offsets. Python can list the files from
    the central directory, but extraction seeks to offsets that are too large by
    exactly 2^32 bytes. Correct that wraparound while avoiding raw signature
    scans, which can match bytes inside compressed payloads.
    """
    sig = b"PK\x03\x04"
    zip32_wrap = 2**32
    headers = []
    with zipfile.ZipFile(zip_path) as zf, open(zip_path, "rb") as f:
        for info in zf.infolist():
            candidate_offsets = [info.header_offset]
            if info.header_offset >= zip32_wrap:
                candidate_offsets.append(info.header_offset - zip32_wrap)

            parsed_header = None
            for offset in candidate_offsets:
                if offset < 0:
                    continue
                f.seek(offset)
                header = f.read(30)
                if len(header) != 30 or header[:4] != sig:
                    continue
                name_len = int.from_bytes(header[26:28], "little")
                extra_len = int.from_bytes(header[28:30], "little")
                name = f.read(name_len).decode("utf-8", errors="replace")
                if name != info.filename:
                    continue
                parsed_header = (offset, header, name_len, extra_len, name)
                break

            if parsed_header is None:
                raise zipfile.BadZipFile(f"Could not locate local header for {info.filename}")

            offset, header, name_len, extra_len, name = parsed_header
            flags = int.from_bytes(header[6:8], "little")
            method = int.from_bytes(header[8:10], "little")
            data_start = offset + 30 + name_len + extra_len
            headers.append(
                {
                    "offset": offset,
                    "flags": flags,
                    "method": method,
                    "crc": info.CRC,
                    "compressed_size": info.compress_size,
                    "uncompressed_size": info.file_size,
                    "name": name,
                    "data_start": data_start,
                    "data_end": data_start + info.compress_size,
                }
            )

    headers.sort(key=lambda h: h["offset"])
    for i, header in enumerate(headers[:-1]):
        next_offset = headers[i + 1]["offset"]
        if next_offset > header["data_start"]:
            header["data_end"] = next_offset
    return headers


def _safe_output_path(data_dir: Path, name: str) -> Path:
    out_path = data_dir / name
    resolved_base = data_dir.resolve()
    resolved_out = out_path.resolve()
    if resolved_base not in resolved_out.parents and resolved_base != resolved_out:
        raise ValueError(f"Unsafe zip member path: {name}")
    return out_path


def _copy_stored_member(f, start: int, end: int, out_path: Path):
    f.seek(start)
    remaining = end - start
    with open(out_path, "wb") as out:
        while remaining > 0:
            chunk = f.read(min(1024 * 1024, remaining))
            if not chunk:
                raise EOFError(f"Unexpected EOF while extracting {out_path}")
            out.write(chunk)
            remaining -= len(chunk)


def _inflate_deflated_member(f, start: int, end: int, out_path: Path):
    f.seek(start)
    remaining = end - start
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    with open(out_path, "wb") as out:
        while remaining > 0 and not decompressor.eof:
            chunk = f.read(min(1024 * 1024, remaining))
            if not chunk:
                raise EOFError(f"Unexpected EOF while extracting {out_path}")
            remaining -= len(chunk)
            out.write(decompressor.decompress(chunk))
        out.write(decompressor.flush())
    if not decompressor.eof:
        raise zipfile.BadZipFile(f"Could not find end of deflate stream for {out_path}")


def _extract_malformed_zip(zip_path: Path, data_dir: Path):
    print("Falling back to local-header extraction for malformed zip.")
    headers = _local_zip_headers(zip_path)
    with open(zip_path, "rb") as f:
        for header in headers:
            name = header["name"]
            if name.startswith("__MACOSX/") or "/._" in name or name.startswith("._"):
                continue
            out_path = _safe_output_path(data_dir, name)
            if name.endswith("/"):
                out_path.mkdir(parents=True, exist_ok=True)
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"  Extracting {name}")
            if header["method"] == 0:
                _copy_stored_member(f, header["data_start"], header["data_end"], out_path)
            elif header["method"] == 8:
                _inflate_deflated_member(f, header["data_start"], header["data_end"], out_path)
            else:
                raise zipfile.BadZipFile(f"Unsupported compression method {header['method']} for {name}")


def _extract_zip(zip_path: Path, data_dir: Path):
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)
    except zipfile.BadZipFile as exc:
        print(f"Standard zip extraction failed: {exc}")
        _extract_malformed_zip(zip_path, data_dir)


def _download_valid_zip(file_info: dict, zip_path: Path, retries: int) -> bool:
    for attempt_i in range(retries + 1):
        if zip_path.exists() and zip_path.stat().st_size == file_info["size"]:
            print(f"Validating existing download: {zip_path}")
            if _zip_is_valid(zip_path):
                return True
            if _zip_can_be_listed(zip_path):
                print(
                    "Existing zip failed strict validation but has a readable file list; "
                    "will try fallback extraction."
                )
                return True
            print(f"Existing zip is corrupt and unreadable; deleting {zip_path}")
            zip_path.unlink()
        elif zip_path.exists():
            print(
                f"Existing download has unexpected size "
                f"({zip_path.stat().st_size} != {file_info['size']}); redownloading."
            )
            zip_path.unlink()

        print(f"Downloading {file_info['name']} (attempt {attempt_i + 1}/{retries + 1})...")
        _download_file(file_info["download_url"], zip_path, expected_size=file_info["size"])
        if zip_path.stat().st_size != file_info["size"]:
            print(
                f"Downloaded size mismatch for {zip_path}: "
                f"{zip_path.stat().st_size} != {file_info['size']}"
            )
            continue
        print(f"Validating downloaded zip: {zip_path}")
        if _zip_is_valid(zip_path):
            return True
        if _zip_can_be_listed(zip_path):
            print(
                "Downloaded zip failed strict validation but has a readable file list; "
                "will try fallback extraction."
            )
            return True
        print(f"Downloaded zip failed validation and is unreadable: {zip_path}")

    return False


def _normalize_subject_9(data_dir: Path):
    subject_09 = data_dir / "subject_09"
    subject_9 = data_dir / "subject_9"
    if subject_09.exists() and not subject_9.exists():
        subject_09.rename(subject_9)

    root_subject_9_files = [
        "data_full.mat",
        "data_full_050617.mat",
        "TimeSeries.h5",
        "TimeSeries_half.h5",
        "README_LICENSE.rtf",
    ]
    if (data_dir / "data_full.mat").exists() and (data_dir / "TimeSeries.h5").exists():
        subject_9.mkdir(parents=True, exist_ok=True)
        for file_name in root_subject_9_files:
            src = data_dir / file_name
            dst = subject_9 / file_name
            if src.exists() and not dst.exists():
                src.rename(dst)


def _verify_subject(data_dir: Path, subject: int, required_files: list[str]) -> list[str]:
    subject_dir = data_dir / _subject_dir_name(subject)
    missing = []
    for file_name in required_files:
        if not (subject_dir / file_name).exists():
            missing.append(str(subject_dir / file_name))
    return missing


def _verify_artifacts(data_dir: Path) -> list[str]:
    artifact_dir = data_dir / "fold_and_segment_structures"
    return [str(artifact_dir / file_name) for file_name in REQUIRED_ARTIFACT_FILES
            if not (artifact_dir / file_name).exists()]


def _verify_additional_mat_files(data_dir: Path) -> list[str]:
    additional_dir = data_dir / "Additional_mat_files"
    return [str(additional_dir / file_name) for file_name in ADDITIONAL_MAT_FILES
            if not (additional_dir / file_name).exists()]


def _additional_mat_files_are_complete(data_dir: Path) -> bool:
    return len(_verify_additional_mat_files(data_dir)) == 0


def _generate_artifacts(data_dir: Path):
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "data/generate_fold_and_segment_artifacts.py"),
            "--data-dir",
            str(data_dir),
            "--artifact-dir",
            str(data_dir / "fold_and_segment_structures"),
        ],
        check=True,
    )


def _subject_download_is_complete(data_dir: Path, subject: int) -> bool:
    return len(_verify_subject(data_dir, subject, DOWNLOADED_SUBJECT_FILES)) == 0


def _file_manifest(article: dict) -> dict[str, dict]:
    return {file_info["name"]: file_info for file_info in article["files"]}


def _format_size(n_bytes: int) -> str:
    gb = n_bytes / (1024 ** 3)
    return f"{gb:.2f} GB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Directory where subject folders should be created.",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SUBJECTS),
        help="Comma-separated subject ids to download.",
    )
    additional_mat_group = parser.add_mutually_exclusive_group()
    additional_mat_group.add_argument(
        "--include-additional-mat-files",
        dest="include_additional_mat_files",
        action="store_true",
        help="Download Additional_mat_files.zip, which includes shared reference files.",
    )
    additional_mat_group.add_argument(
        "--no-additional-mat-files",
        dest="include_additional_mat_files",
        action="store_false",
        help="Do not download Additional_mat_files.zip.",
    )
    parser.set_defaults(include_additional_mat_files=True)
    parser.add_argument(
        "--keep-zips",
        action="store_true",
        help="Keep downloaded zip files after extraction.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned downloads without downloading data.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of download retries after a corrupt or incomplete zip.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not download; only check whether required files are present.",
    )
    artifact_generation_group = parser.add_mutually_exclusive_group()
    artifact_generation_group.add_argument(
        "--generate-artifacts",
        dest="generate_artifacts",
        action="store_true",
        help=(
            "Generate missing JSON fold/segment artifacts from downloaded subject data. "
            "This is the default unless --no-generate-artifacts is passed."
        ),
    )
    artifact_generation_group.add_argument(
        "--no-generate-artifacts",
        dest="generate_artifacts",
        action="store_false",
        help="Do not generate missing fold/segment JSON artifacts.",
    )
    parser.set_defaults(generate_artifacts=True)
    parser.add_argument(
        "--no-artifact-check",
        action="store_true",
        help="Skip verification of fold/segment JSON artifacts.",
    )
    args = parser.parse_args()

    subjects = [int(v.strip()) for v in args.subjects.split(",") if v.strip()]
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        _normalize_subject_9(data_dir)
        missing_subject_files = []
        for subject in subjects:
            missing_subject_files.extend(_verify_subject(data_dir, subject, ANALYSIS_READY_SUBJECT_FILES))
        if args.include_additional_mat_files:
            missing_subject_files.extend(_verify_additional_mat_files(data_dir))
        missing_artifacts = [] if args.no_artifact_check else _verify_artifacts(data_dir)
        if missing_artifacts and args.generate_artifacts and not missing_subject_files:
            print("Generating missing fold/segment JSON artifacts...")
            _generate_artifacts(data_dir)
            missing_artifacts = _verify_artifacts(data_dir)
        missing = missing_subject_files + missing_artifacts
        if missing:
            print("Missing required files:")
            for path in missing:
                print(f"  {path}")
            return 1
        print("All required subject and fold/segment artifact files are present.")
        return 0

    if not args.dry_run:
        _normalize_subject_9(data_dir)
        missing_local_files = []
        for subject in subjects:
            missing_local_files.extend(_verify_subject(data_dir, subject, ANALYSIS_READY_SUBJECT_FILES))
        if args.include_additional_mat_files:
            missing_local_files.extend(_verify_additional_mat_files(data_dir))

        if not missing_local_files:
            artifact_missing = [] if args.no_artifact_check else _verify_artifacts(data_dir)
            if artifact_missing and args.generate_artifacts:
                print("All requested data files are present.")
                print("Generating missing fold/segment JSON artifacts...")
                _generate_artifacts(data_dir)
                artifact_missing = _verify_artifacts(data_dir)
            if not artifact_missing:
                print("All requested data and fold/segment artifact files are present.")
                return 0
            print("Subject data are present, but fold/segment artifacts are missing.", file=sys.stderr)
            print("Generate them with:", file=sys.stderr)
            print(
                f"  python data/generate_fold_and_segment_artifacts.py --data-dir {data_dir}",
                file=sys.stderr,
            )
            print("Missing artifact files:", file=sys.stderr)
            for path in artifact_missing:
                print(f"  {path}", file=sys.stderr)
            return 1

    article = _fetch_article_metadata()
    manifest = _file_manifest(article)
    requested_items = [(_subject_file_name(subject), subject) for subject in subjects]
    if args.include_additional_mat_files:
        requested_items.append(("Additional_mat_files.zip", None))
    requested_files = [name for name, _ in requested_items]

    missing_manifest = [name for name in requested_files if name not in manifest]
    if missing_manifest:
        print("The Figshare article did not contain expected files:", file=sys.stderr)
        for name in missing_manifest:
            print(f"  {name}", file=sys.stderr)
        return 1

    total_bytes = sum(manifest[name]["size"] for name in requested_files)
    print(f"Article: {article['title']}")
    print(f"DOI: {article['doi']}")
    print(f"Destination: {data_dir.resolve()}")
    print(f"Requested files: {len(requested_files)}")
    print(f"Total download size: {_format_size(total_bytes)}")
    for name in requested_files:
        print(f"  {name}: {_format_size(manifest[name]['size'])}")

    if args.dry_run:
        return 0

    for name, subject in requested_items:
        file_info = manifest[name]
        zip_path = data_dir / name

        if subject is not None:
            _normalize_subject_9(data_dir)
            if _subject_download_is_complete(data_dir, subject):
                print(f"Subject {subject} already present; skipping {name}.")
                continue
        elif name == "Additional_mat_files.zip" and _additional_mat_files_are_complete(data_dir):
            print(f"Additional MAT files already present; skipping {name}.")
            continue

        if not _download_valid_zip(file_info=file_info, zip_path=zip_path, retries=args.retries):
            print(f"Failed to download a valid zip after retries: {name}", file=sys.stderr)
            return 1

        print(f"Extracting {name}...")
        _extract_zip(zip_path, data_dir)
        _normalize_subject_9(data_dir)
        if not args.keep_zips:
            zip_path.unlink()

    _normalize_subject_9(data_dir)

    missing = []
    for subject in subjects:
        missing.extend(_verify_subject(data_dir, subject, ANALYSIS_READY_SUBJECT_FILES))
    if args.include_additional_mat_files:
        missing.extend(_verify_additional_mat_files(data_dir))
    if missing:
        print("Download/extraction completed, but analysis-ready files are missing.", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 1

    artifact_missing = [] if args.no_artifact_check else _verify_artifacts(data_dir)
    if artifact_missing and args.generate_artifacts:
        print("Generating missing fold/segment JSON artifacts...")
        _generate_artifacts(data_dir)
        artifact_missing = _verify_artifacts(data_dir)

    if artifact_missing:
        print("Subject data are present, but fold/segment artifacts are missing.", file=sys.stderr)
        print("Generate them with:", file=sys.stderr)
        print(
            f"  python data/generate_fold_and_segment_artifacts.py --data-dir {data_dir}",
            file=sys.stderr,
        )
        print("Missing artifact files:", file=sys.stderr)
        for path in artifact_missing:
            print(f"  {path}", file=sys.stderr)
        return 1

    print("Data download and verification completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
