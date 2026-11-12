#!/usr/bin/env python3
"""Convert FineVision parquet subsets to memory-mappable Arrow IPC with a per-subset index.

``convert`` writes each ``<subset>/train-*.parquet`` as ``<subset>/<name>.arrow``,
one record batch per row group so rows keep their order, plus a ``<name>.meta.npz``
sidecar (rows per batch, per-row keep flag, image count, ratings and the size of
the source file). Each file is claimed with an exclusive ``.claim`` file first, so
any number of ``convert`` processes can share the work and a rerun skips finished
files; a claim is touched after every row group, and one older than
``--stale-claim-hours`` is taken over, so each process writes through temp files
named for it and releases only a claim it still holds. A file that fails to
convert leaves no temp files, the other files go on, and the failures are listed
at the end with a non-zero exit. ``index`` merges a fully converted subset's
sidecars into ``<subset>/index.npz``, refusing a subset with a file not converted
or whose source changed size since its conversion; for a sidecar without image
counts it reads them from the Arrow file.

    uv run python examples/image_understanding/scripts/convert_finevision_to_arrow.py convert \\
        --src /path/to/FineVision --dst /path/to/FineVision-arrow --procs 8
    uv run python examples/image_understanding/scripts/convert_finevision_to_arrow.py index \\
        --src /path/to/FineVision --dst /path/to/FineVision-arrow
"""

from __future__ import annotations

import argparse
import os
import random
import socket
import sys
import time
from collections.abc import Callable
from multiprocessing import Pool
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
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


def keep_flags(texts: pa.Array) -> np.ndarray:
    """Which conversations open with a turn holding both a user message and a reply.

    The reader keeps a conversation up to its first incomplete turn, so a row
    without a complete first turn has nothing to train.
    """
    offsets = texts.offsets.to_numpy()
    turns = texts.values

    def present(name: str) -> np.ndarray:
        text = pc.fill_null(turns.field(name).cast(pa.string()), "")
        length = pc.utf8_length(pc.utf8_trim_whitespace(text))
        return pc.greater(length, 0).to_numpy(zero_copy_only=False)

    complete = np.concatenate([present("user") & present("assistant"), [False]])
    starts, ends = offsets[:-1], offsets[1:]
    flags = (ends > starts) & complete[np.minimum(starts, len(complete) - 1)]
    if texts.null_count:
        flags &= ~texts.is_null().to_numpy(zero_copy_only=False)
    return flags


def image_counts(batch: pa.RecordBatch) -> np.ndarray:
    """How many images each row holds; a batch without an images column holds none."""
    names = batch.schema.names
    if "images" not in names or pa.types.is_null(batch.schema.field("images").type):
        return np.zeros(batch.num_rows, dtype=np.int16)
    lengths = batch.column("images").value_lengths().fill_null(0)
    return lengths.to_numpy(zero_copy_only=False).astype(np.int16)


def convert_file(
    src_file: Path, dst_file: Path, tag: str = "", heartbeat: Callable[[], None] | None = None
) -> int:
    """Write ``src_file`` as Arrow IPC and its sidecar through temp files named by ``tag``.

    ``heartbeat`` is called after every row group. A failure removes the temp files.
    """
    tmp = dst_file.with_name(f"{dst_file.name}.{tag}.tmp")
    meta_tmp = dst_file.with_name(f"{dst_file.stem}.meta.{tag}.tmp.npz")
    try:
        with open(src_file, "rb") as source:
            return _convert_file(source, src_file, dst_file, tmp, meta_tmp, heartbeat)
    except BaseException:
        tmp.unlink(missing_ok=True)
        meta_tmp.unlink(missing_ok=True)
        raise


def _convert_file(
    source: BinaryIO,
    src_file: Path,
    dst_file: Path,
    tmp: Path,
    meta_tmp: Path,
    heartbeat: Callable[[], None] | None,
) -> int:
    source_size = os.fstat(source.fileno()).st_size
    parquet = pq.ParquetFile(source)
    batch_rows: list[int] = []
    keep: list[np.ndarray] = []
    images: list[np.ndarray] = []
    ratings: dict[str, list[np.ndarray]] = {name: [] for name in RATING_COLUMNS}
    with pa.OSFile(str(tmp), "wb") as sink, pa.ipc.new_file(sink, parquet.schema_arrow) as writer:
        for group in range(parquet.num_row_groups):
            table = parquet.read_row_group(group).combine_chunks()
            batches = table.to_batches() or [pa.RecordBatch.from_pylist([], schema=table.schema)]
            if len(batches) != 1:
                raise RuntimeError(f"{src_file} row group {group} became {len(batches)} batches")
            writer.write_batch(batches[0])
            batch_rows.append(table.num_rows)
            keep.append(keep_flags(batches[0].column("texts")))
            images.append(image_counts(batches[0]))
            for name in RATING_COLUMNS:
                if name in table.column_names:
                    column = pc.fill_null(table.column(name), -1).to_numpy()
                    ratings[name].append(column.astype(np.int16))
                else:
                    ratings[name].append(np.full(table.num_rows, -1, dtype=np.int16))
            if heartbeat is not None:
                heartbeat()
    reader = pa.ipc.open_file(pa.memory_map(str(tmp), "r"))
    if reader.num_record_batches != parquet.num_row_groups:
        raise RuntimeError(f"{tmp} holds {reader.num_record_batches} batches")
    if not reader.schema.equals(parquet.schema_arrow):
        raise RuntimeError(f"{tmp} schema differs from {src_file}")
    for group, rows in enumerate(batch_rows):
        if reader.get_batch(group).num_rows != rows:
            raise RuntimeError(f"{tmp} batch {group} does not hold {rows} rows")
    os.replace(tmp, dst_file)
    np.savez(
        meta_tmp,
        batch_rows=np.asarray(batch_rows, dtype=np.int64),
        keep=np.concatenate(keep) if keep else np.zeros(0, dtype=bool),
        num_images=np.concatenate(images) if images else np.zeros(0, dtype=np.int16),
        source_size=np.int64(source_size),
        **{
            name: np.concatenate(values) if values else np.zeros(0, dtype=np.int16)
            for name, values in ratings.items()
        },
    )
    os.replace(meta_tmp, meta_path(dst_file))
    return dst_file.stat().st_size


class ClaimLostError(Exception):
    """Another process took this file over."""


def _touch_claim(claim: Path, tag: str) -> None:
    """Refresh ``claim``, or raise ``ClaimLostError`` when it is gone or another's."""
    try:
        if not claim.read_text().startswith(f"{tag} "):
            raise ClaimLostError(f"{claim} is held by another process")
        os.utime(claim)
    except FileNotFoundError:
        raise ClaimLostError(f"{claim} was taken over") from None


def _release_claim(claim: Path, tag: str) -> None:
    """Remove ``claim`` when it is still this process's, leaving another's untouched."""
    try:
        if claim.read_text().startswith(f"{tag} "):
            claim.unlink()
    except FileNotFoundError:
        pass


def _convert_job(job: tuple[Path, Path, float]) -> int | str:
    """The bytes written for one file, 0 when it is done or claimed, or the failure.

    A stale claim is taken over by renaming it, which only one process can do; a
    process whose claim was taken over stops at its next row group and leaves the
    file and its claim to the taker. An error while claiming is this file's failure.
    """
    src_file, dst_file, stale_seconds = job
    claim = dst_file.with_name(dst_file.name + ".claim")
    tag = f"{socket.gethostname()}-{os.getpid()}"
    if meta_path(dst_file).exists() and dst_file.exists():
        return 0
    stale = claim.with_name(f"{claim.name}.stale-{tag}")
    try:
        try:
            if time.time() - claim.stat().st_mtime > stale_seconds:
                os.rename(claim, stale)
                stale.unlink()
        except FileNotFoundError:
            pass
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return 0
    except OSError as exc:
        return f"{src_file}: {type(exc).__name__}: {exc}"
    lost = False
    try:
        try:
            os.write(fd, f"{tag} {time.time():.0f}\n".encode())
        finally:
            os.close(fd)
        return convert_file(src_file, dst_file, tag, heartbeat=lambda: _touch_claim(claim, tag))
    except ClaimLostError:
        lost = True
        return 0
    except Exception as exc:
        return f"{src_file}: {type(exc).__name__}: {exc}"
    finally:
        if not lost:
            _release_claim(claim, tag)


def convert(args: argparse.Namespace) -> None:
    jobs = []
    failures: list[str] = []
    for subset in args.subsets or subsets(args.src):
        files = source_files(args.src, subset)
        if not files:
            failures.append(f"{subset}: no train-*.parquet files under {args.src / subset}")
            continue
        (args.dst / subset).mkdir(parents=True, exist_ok=True)
        jobs += [(f, arrow_path(args.dst, f), args.stale_claim_hours * 3600) for f in files]
    random.Random(f"{socket.gethostname()}-{os.getpid()}").shuffle(jobs)
    start, total, done = time.time(), 0, 0
    with Pool(args.procs) as pool:
        for result in pool.imap_unordered(_convert_job, jobs):
            if isinstance(result, str):
                failures.append(result)
            elif result:
                done += 1
                total += result
                if done % 20 == 0:
                    rate = total / (time.time() - start) / 2**20
                    print(f"{done} files, {total / 2**30:.0f} GiB, {rate:.0f} MiB/s", flush=True)
    print(f"converted {done} files, {total / 2**30:.1f} GiB in {time.time() - start:.0f} s")
    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        sys.exit(f"{len(failures)} files failed to convert")


def _file_image_counts(arrow_file: Path) -> np.ndarray:
    reader = pa.ipc.open_file(pa.memory_map(str(arrow_file), "r"))
    counts = [image_counts(reader.get_batch(i)) for i in range(reader.num_record_batches)]
    return np.concatenate(counts) if counts else np.zeros(0, dtype=np.int16)


def _subset_index(job: tuple[Path, Path, str]) -> str:
    src, dst, subset = job
    sources = source_files(src, subset)
    if not sources:
        return f"{subset}: no train-*.parquet files under {src / subset}"
    arrows = [arrow_path(dst, f) for f in sources]
    missing = [a for a in arrows if not (a.exists() and meta_path(a).exists())]
    if missing:
        return f"{subset}: {len(missing)} of {len(arrows)} files not converted yet"
    metas = [dict(np.load(meta_path(a))) for a in arrows]
    changed = [
        source
        for source, meta in zip(sources, metas, strict=True)
        if "source_size" in meta and int(meta["source_size"]) != source.stat().st_size
    ]
    if changed:
        return (
            f"{subset}: {len(changed)} files changed size since their conversion; "
            "remove their .meta.npz and convert again"
        )
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
        print(problem, file=sys.stderr)
    if problems:
        sys.exit(f"{len(problems)} subsets could not be indexed")
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
