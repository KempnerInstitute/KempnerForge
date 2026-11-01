#!/usr/bin/env python3
"""Convert FineVision parquet subsets to memory-mappable Arrow IPC with a per-subset index.

``convert`` writes each ``<subset>/train-*.parquet`` as ``<subset>/<name>.arrow``,
one record batch per row group so rows keep their order, plus a ``<name>.meta.npz``
sidecar (rows per batch, per-row keep flag, image count and ratings). Each file is
claimed with an exclusive ``.claim`` file first, so any number of ``convert``
processes can share the work and a rerun skips finished files. ``index`` merges a
fully converted subset's sidecars into ``<subset>/index.npz``; for a sidecar
without image counts it reads them from the Arrow file.

    uv run python examples/vlm/scripts/convert_finevision_to_arrow.py convert \\
        --src /path/to/FineVision --dst /path/to/FineVision-arrow --procs 8
    uv run python examples/vlm/scripts/convert_finevision_to_arrow.py index \\
        --src /path/to/FineVision --dst /path/to/FineVision-arrow
"""

from __future__ import annotations

import argparse
import os
import random
import socket
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RATING_COLUMNS = (
    "relevance_min",
    "visual_dependency_min",
    "image_correspondence_min",
    "formatting_min",
)


def source_files(src: Path, subset: str) -> list[Path]:
    """A subset's parquet files in the order its rows are numbered."""
    return sorted((src / subset).glob("train-*.parquet"))


def subsets(src: Path) -> list[str]:
    """Every directory of ``src`` that holds parquet files."""
    return sorted(d.name for d in src.iterdir() if d.is_dir() and source_files(src, d.name))


def arrow_path(dst: Path, parquet_file: Path) -> Path:
    return dst / parquet_file.parent.name / (parquet_file.stem + ".arrow")


def meta_path(arrow_file: Path) -> Path:
    return arrow_file.with_name(arrow_file.stem + ".meta.npz")


def keep_row(turns: list[dict] | None) -> bool:
    """A conversation is kept when it has turns and its last answer is non-empty."""
    return bool(turns) and bool(turns[-1].get("assistant"))


def image_counts(batch: pa.RecordBatch) -> np.ndarray:
    lengths = batch.column("images").value_lengths().fill_null(0)
    return lengths.to_numpy(zero_copy_only=False).astype(np.int16)


def convert_file(src_file: Path, dst_file: Path) -> int:
    """Write ``src_file`` as Arrow IPC and its sidecar; return the bytes written."""
    parquet = pq.ParquetFile(src_file)
    tmp = dst_file.with_name(dst_file.name + ".tmp")
    batch_rows: list[int] = []
    keep: list[bool] = []
    images: list[np.ndarray] = []
    ratings: dict[str, list[int]] = {name: [] for name in RATING_COLUMNS}
    with pa.OSFile(str(tmp), "wb") as sink, pa.ipc.new_file(sink, parquet.schema_arrow) as writer:
        for group in range(parquet.num_row_groups):
            table = parquet.read_row_group(group).combine_chunks()
            batches = table.to_batches()
            if len(batches) != 1:
                raise RuntimeError(f"{src_file} row group {group} became {len(batches)} batches")
            writer.write_batch(batches[0])
            batch_rows.append(table.num_rows)
            keep.extend(keep_row(turns) for turns in table.column("texts").to_pylist())
            images.append(image_counts(batches[0]))
            for name in RATING_COLUMNS:
                values = (
                    table.column(name).to_pylist()
                    if name in table.column_names
                    else [None] * table.num_rows
                )
                ratings[name].extend(-1 if v is None else v for v in values)
    reader = pa.ipc.open_file(pa.memory_map(str(tmp), "r"))
    if reader.num_record_batches != parquet.num_row_groups:
        raise RuntimeError(f"{tmp} holds {reader.num_record_batches} batches")
    if not reader.schema.equals(parquet.schema_arrow):
        raise RuntimeError(f"{tmp} schema differs from {src_file}")
    for group, rows in enumerate(batch_rows):
        if reader.get_batch(group).num_rows != rows:
            raise RuntimeError(f"{tmp} batch {group} does not hold {rows} rows")
    os.replace(tmp, dst_file)
    meta_tmp = dst_file.with_name(dst_file.stem + ".meta.tmp.npz")
    np.savez(
        meta_tmp,
        batch_rows=np.asarray(batch_rows, dtype=np.int64),
        keep=np.asarray(keep, dtype=bool),
        num_images=np.concatenate(images) if images else np.zeros(0, dtype=np.int16),
        **{name: np.asarray(values, dtype=np.int16) for name, values in ratings.items()},
    )
    os.replace(meta_tmp, meta_path(dst_file))
    return dst_file.stat().st_size


def _convert_job(job: tuple[Path, Path, float]) -> int:
    src_file, dst_file, stale_seconds = job
    claim = dst_file.with_name(dst_file.name + ".claim")
    if meta_path(dst_file).exists():
        return 0
    try:
        if time.time() - claim.stat().st_mtime > stale_seconds:
            claim.unlink()
    except FileNotFoundError:
        pass
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return 0
    os.write(fd, f"{socket.gethostname()} {os.getpid()} {time.time():.0f}\n".encode())
    os.close(fd)
    try:
        return convert_file(src_file, dst_file)
    finally:
        claim.unlink()


def convert(args: argparse.Namespace) -> None:
    jobs = []
    for subset in args.subsets or subsets(args.src):
        (args.dst / subset).mkdir(parents=True, exist_ok=True)
        jobs += [
            (f, arrow_path(args.dst, f), args.stale_claim_hours * 3600)
            for f in source_files(args.src, subset)
        ]
    random.Random(f"{socket.gethostname()}-{os.getpid()}").shuffle(jobs)
    start, total, done = time.time(), 0, 0
    with Pool(args.procs) as pool:
        for written in pool.imap_unordered(_convert_job, jobs):
            if written:
                done += 1
                total += written
                if done % 20 == 0:
                    rate = total / (time.time() - start) / 2**20
                    print(f"{done} files, {total / 2**30:.0f} GiB, {rate:.0f} MiB/s", flush=True)
    print(f"converted {done} files, {total / 2**30:.1f} GiB in {time.time() - start:.0f} s")


def _file_image_counts(arrow_file: Path) -> np.ndarray:
    reader = pa.ipc.open_file(pa.memory_map(str(arrow_file), "r"))
    counts = [image_counts(reader.get_batch(i)) for i in range(reader.num_record_batches)]
    return np.concatenate(counts) if counts else np.zeros(0, dtype=np.int16)


def _subset_index(job: tuple[Path, Path, str]) -> str:
    src, dst, subset = job
    sources = source_files(src, subset)
    arrows = [arrow_path(dst, f) for f in sources]
    missing = [a for a in arrows if not meta_path(a).exists()]
    if missing:
        return f"{subset}: {len(missing)} of {len(arrows)} files not converted yet"
    metas = [dict(np.load(meta_path(a))) for a in arrows]
    for meta, arrow in zip(metas, arrows, strict=True):
        if "num_images" not in meta:
            meta["num_images"] = _file_image_counts(arrow)
    tmp = dst / subset / "index.tmp.npz"
    np.savez(
        tmp,
        files=np.asarray([a.name for a in arrows]),
        file_batches=np.asarray([len(m["batch_rows"]) for m in metas], dtype=np.int64),
        batch_rows=np.concatenate([m["batch_rows"] for m in metas]),
        keep=np.concatenate([m["keep"] for m in metas]),
        num_images=np.concatenate([m["num_images"] for m in metas]),
        source_sizes=np.asarray([f.stat().st_size for f in sources], dtype=np.int64),
        **{name: np.concatenate([m[name] for m in metas]) for name in RATING_COLUMNS},
    )
    os.replace(tmp, dst / subset / "index.npz")
    return ""


def index(args: argparse.Namespace) -> None:
    chosen = args.subsets or subsets(args.src)
    with Pool(args.procs) as pool:
        problems = [
            p for p in pool.imap(_subset_index, [(args.src, args.dst, s) for s in chosen]) if p
        ]
    for problem in problems:
        print(problem)
    if problems:
        sys.exit(f"{len(problems)} subsets are not fully converted; no index written for them")
    print(f"wrote index.npz for {len(chosen)} subsets")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["convert", "index"])
    parser.add_argument("--src", type=Path, required=True, help="FineVision parquet root")
    parser.add_argument("--dst", type=Path, required=True, help="Arrow output root")
    parser.add_argument("--procs", type=int, default=2)
    parser.add_argument("--subsets", nargs="*", help="only these subsets (default: all)")
    parser.add_argument("--stale-claim-hours", type=float, default=6.0)
    args = parser.parse_args()
    {"convert": convert, "index": index}[args.command](args)


if __name__ == "__main__":
    main()
