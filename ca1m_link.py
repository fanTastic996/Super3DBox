#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DIGITS_RE = re.compile(r"(\d+)")  # grab any digits in filename


def parse_seq_name(j: dict, json_path: Path) -> str:
    """
    Priority:
      1) json["seq_dir"] basename (matches your example)
      2) json filename stem
    """
    seq_dir = j.get("seq_dir", "")
    if isinstance(seq_dir, str) and seq_dir.strip():
        return Path(seq_dir).name
    return json_path.stem


def build_frame_map(rgb_dir: Path) -> Dict[int, Path]:
    """
    Build mapping: frame_id(int) -> file path
    Extract digits from filename (before extension or anywhere), take the LAST digit group.
    Example:
      00013.jpg -> 13
      12312.png -> 12312
      DSC01752.JPG -> 1752
    If multiple files map to same id, keep the lexicographically smallest path for determinism.
    """
    frame_map: Dict[int, Path] = {}
    # list all files (ignore directories)
    files = [Path(p) for p in glob.glob(str(rgb_dir / "*")) if Path(p).is_file()]

    for fp in files:
        name = fp.name
        groups = DIGITS_RE.findall(name)
        if not groups:
            continue
        fid = int(groups[-1])  # last digit group tends to be the frame index
        if fid not in frame_map or str(fp) < str(frame_map[fid]):
            frame_map[fid] = fp
    return frame_map


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_symlink(src: Path, dst: Path, force: bool = False, relative: bool = True) -> Tuple[bool, str]:
    """
    Create a symlink dst -> src.
    Returns (success, message).
    """
    try:
        if dst.exists() or dst.is_symlink():
            if force:
                dst.unlink()
            else:
                return True, f"exists: {dst}"
        ensure_dir(dst.parent)

        link_target = os.path.relpath(src, start=dst.parent) if relative else str(src)
        os.symlink(link_target, dst)
        return True, f"linked: {dst} -> {link_target}"
    except Exception as e:
        return False, f"FAILED: {dst} ({e})"


def main():
    ap = argparse.ArgumentParser(description="Create benchmark symlink folders from CA-1M benchmark jsons.")
    ap.add_argument("--json_dir", type=str, required=True, help="Directory containing benchmark json files (e.g., 107 jsons).")
    ap.add_argument("--root_dir", type=str, required=True, help="Dataset root dir containing <seq>/rgb/ folders.")
    ap.add_argument("--target_dir", type=str, required=True, help="Output directory to create <seq>/<group>/images symlinks.")
    ap.add_argument("--rgb_subdir", type=str, default="rgb", help="Subfolder name under each seq (default: rgb).")
    ap.add_argument("--force", action="store_true", help="Overwrite existing links/files.")
    ap.add_argument("--absolute", action="store_true", help="Use absolute symlink paths (default: relative).")
    ap.add_argument("--dry_run", action="store_true", help="Print actions but do not create links.")
    args = ap.parse_args()

    json_dir = Path(args.json_dir)
    root_dir = Path(args.root_dir)
    target_dir = Path(args.target_dir)

    json_paths = sorted(json_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No .json found in {json_dir}")

    total_json = 0
    total_links = 0
    total_missing = 0

    for jp in json_paths:
        total_json += 1
        with jp.open("r", encoding="utf-8") as f:
            j = json.load(f)

        seq_name = parse_seq_name(j, jp)
        rgb_dir = root_dir / seq_name / args.rgb_subdir
        if not rgb_dir.exists():
            print(f"[WARN] rgb_dir not found: {rgb_dir} (skip {seq_name} from {jp})")
            continue

        frame_map = build_frame_map(rgb_dir)
        if not frame_map:
            print(f"[WARN] no images indexed under {rgb_dir} (skip {seq_name})")
            continue

        windows: List[dict] = j.get("windows", [])
        if not isinstance(windows, list) or len(windows) == 0:
            print(f"[WARN] json has no 'windows': {jp} (skip)")
            continue

        # We assume 10 groups; but will just enumerate whatever is there
        for group_idx, w in enumerate(windows):
            frame_ids = w.get("frame_ids", [])
            if not isinstance(frame_ids, list):
                print(f"[WARN] bad frame_ids in {jp} window#{group_idx} (skip window)")
                continue

            out_images_dir = target_dir / seq_name / str(group_idx) / "images"
            if not args.dry_run:
                ensure_dir(out_images_dir)

            for fid in frame_ids:
                try:
                    fid_int = int(fid)
                except Exception:
                    print(f"[WARN] non-int frame id {fid} in {jp} window#{group_idx}")
                    total_missing += 1
                    continue

                src = frame_map.get(fid_int)
                if src is None:
                    print(f"[MISS] {seq_name} window#{group_idx}: frame_id={fid_int} not found in {rgb_dir}")
                    total_missing += 1
                    continue

                # Keep original filename to avoid extension/padding assumptions
                dst = out_images_dir / src.name

                if args.dry_run:
                    print(f"[DRY] ln -s {'(abs)' if args.absolute else '(rel)'} {src} -> {dst}")
                    total_links += 1
                else:
                    ok, msg = safe_symlink(src, dst, force=args.force, relative=(not args.absolute))
                    if not ok:
                        print(msg)
                    else:
                        total_links += 1

    print("\n===== DONE =====")
    print(f"json processed: {total_json}")
    print(f"symlinks planned/created: {total_links}")
    print(f"missing frames: {total_missing}")


if __name__ == "__main__":
    main()
    
    
'''
python ca1m_link.py \
  --json_dir /data1/lyq/CA-1M-benchmark/ \
  --root_dir /data1/lyq/CA1M-dataset/CA1M-dataset/test/ \
  --target_dir /data1/lyq/CA1M_benchmark_data/ \
  --force
'''