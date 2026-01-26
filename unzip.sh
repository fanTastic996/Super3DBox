#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/lyq/scannetpp_train"
JOBS="${JOBS:-4}"          # 并发数：可 env JOBS=8 ./extract_all_targz.sh
LOG="${LOG:-$ROOT/extract_tar_gz.log}"

echo "[$(date)] Start extracting under: $ROOT" | tee -a "$LOG"
echo "JOBS=$JOBS" | tee -a "$LOG"

# 处理单个 tar.gz 的函数
extract_one() {
  local tarfile="$1"
  local outdir="${tarfile%.tar.gz}"

  # 跳过：如果目标目录存在且非空，认为已解压
  if [[ -d "$outdir" ]] && [[ -n "$(ls -A "$outdir" 2>/dev/null || true)" ]]; then
    echo "[$(date)] SKIP (exists): $tarfile -> $outdir" | tee -a "$LOG"
    return 0
  fi

  mkdir -p "$outdir"
  echo "[$(date)] EXTRACT: $tarfile -> $outdir" | tee -a "$LOG"

  # 解压到同名目录；失败则清理空目录并退出失败
  if tar -xzf "$tarfile" -C "$outdir"; then
    echo "[$(date)] DONE: $tarfile" | tee -a "$LOG"
  else
    echo "[$(date)] FAIL: $tarfile" | tee -a "$LOG"
    rmdir "$outdir" 2>/dev/null || true
    return 1
  fi
}

export -f extract_one
export LOG

# 查找所有 tar.gz 并并发解压
# -print0 / -0 防止文件名含空格出问题
find "$ROOT" -type f -name "*.tar.gz" -print0 \
  | xargs -0 -n 1 -P "$JOBS" bash -c 'extract_one "$@"' _

echo "[$(date)] All done." | tee -a "$LOG"
