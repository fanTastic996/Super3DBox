#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import tarfile
import time
from pathlib import Path

def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    f = float(n)
    for u in units:
        if f < 1024.0:
            return f"{f:.2f}{u}"
        f /= 1024.0
    return f"{f:.2f}EB"

def list_members_safe(t: tarfile.TarFile):
    """
    防止 tar 路径穿越（../../xxx）之类的风险
    """
    for m in t.getmembers():
        p = m.name
        # 禁止绝对路径 & 上跳
        if p.startswith("/") or p.startswith("\\") or ".." in Path(p).parts:
            raise RuntimeError(f"Unsafe path in tar member: {m.name}")
        yield m

def is_done_marker(out_dir: Path, tar_path: Path) -> Path:
    # 每个 tar 解完写一个 marker，便于断点续跑
    return out_dir / f".extract_done__{tar_path.name}"

def extract_one_tar(tar_path: Path, out_dir: Path, *, verbose: bool = True) -> None:
    size = tar_path.stat().st_size
    t0 = time.time()

    if verbose:
        print(f"\n[+] Extract: {tar_path}  ({human_size(size)})")
        print(f"    -> to: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 解压
    with tarfile.open(tar_path, mode="r:*") as t:
        members = list(list_members_safe(t))
        # 有些 tar 里是单个 seq 文件夹（如 42444754/xxx），正好解到 out_dir 下
        t.extractall(path=out_dir, members=members)

    # 简单校验：尝试重新打开 tar 并遍历 member 名称（能读通一般就没问题）
    # 更严格校验（逐文件 hash）太慢不推荐
    with tarfile.open(tar_path, mode="r:*") as t:
        _ = t.getmembers()

    dt = time.time() - t0
    if verbose:
        print(f"[✓] Done: {tar_path.name}  time={dt:.1f}s  speed≈{human_size(int(size / max(dt, 1e-6)))}/s")

def main():
    ap = argparse.ArgumentParser(
        description="Extract all .tar files from a directory to target dir, delete each tar after successful extraction."
    )
    ap.add_argument("--tar_dir", type=str, default="/data1/lyq/CA1M_tar",
                    help="Directory containing many *.tar files.")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Target directory to extract to.")
    ap.add_argument("--pattern", type=str, default="*.tar",
                    help="Glob pattern for tar files, default: *.tar")
    ap.add_argument("--dry_run", action="store_true",
                    help="Only print what would be done, do not extract/delete.")
    ap.add_argument("--skip_done", action="store_true",
                    help="Skip tars that already have a done marker in out_dir.")
    ap.add_argument("--keep_tar", action="store_true",
                    help="Do NOT delete tar after extraction (for debugging).")
    args = ap.parse_args()

    tar_dir = Path(args.tar_dir)
    out_dir = Path(args.out_dir)

    if not tar_dir.is_dir():
        print(f"[ERR] tar_dir not found: {tar_dir}", file=sys.stderr)
        sys.exit(1)

    tars = sorted(tar_dir.glob(args.pattern))
    if not tars:
        print(f"[WARN] No tar files found in {tar_dir} with pattern {args.pattern}")
        return

    print(f"[INFO] tar_dir = {tar_dir}")
    print(f"[INFO] out_dir = {out_dir}")
    print(f"[INFO] found  = {len(tars)} tar files")
    if args.dry_run:
        print("[INFO] dry_run enabled: will not extract/delete")

    ok, fail, skipped = 0, 0, 0

    for i, tar_path in enumerate(tars, 1):
        marker = is_done_marker(out_dir, tar_path)

        if args.skip_done and marker.exists():
            print(f"[=] ({i}/{len(tars)}) Skip (done marker exists): {tar_path.name}")
            skipped += 1
            continue

        print(f"[=] ({i}/{len(tars)}) Processing: {tar_path.name}")

        if args.dry_run:
            print(f"    would extract -> {out_dir}")
            if not args.keep_tar:
                print(f"    would delete  -> {tar_path}")
            print(f"    would write marker -> {marker}")
            continue

        try:
            extract_one_tar(tar_path, out_dir, verbose=True)

            # 写 marker（表示这个 tar 已完成）
            marker.write_text("ok\n", encoding="utf-8")

            # 删除 tar
            if not args.keep_tar:
                tar_path.unlink()
                print(f"[🗑] Deleted: {tar_path.name}")

            ok += 1

        except KeyboardInterrupt:
            print("\n[INTERRUPT] Ctrl-C received. Stop.")
            print(f"[STAT] ok={ok} fail={fail} skipped={skipped}")
            sys.exit(130)

        except Exception as e:
            fail += 1
            print(f"[ERR] Failed: {tar_path.name}\n      {type(e).__name__}: {e}", file=sys.stderr)
            # 失败不删 tar，也不写 marker，方便你重跑
            continue

    print(f"\n[STAT] Finished. ok={ok} fail={fail} skipped={skipped}")

if __name__ == "__main__":
    main()
