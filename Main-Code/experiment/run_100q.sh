#!/bin/bash
#SBATCH -J peft_204_100q
#SBATCH -p gpuA800
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH -o run_100q-%j.out
#SBATCH -e run_100q-%j.err
#
# 100 题主实验:MAB factconsolidation_sh_6k,单历史 + 100 道单跳题。
#
# 两种用法:
#   bash experiment/run_100q.sh            # 在已有 GPU 分配内直接跑(需 CUDA)
#   sbatch experiment/run_100q.sh          # 提交新作业(用上面的 SBATCH 头)
#   bash experiment/run_100q.sh --check    # 只校验配置,不加载模型、不需要 GPU
#
# 可用环境变量覆盖:
#   CONFIG=configs/其它.yaml   RUN_ID=<自定义产物目录名>   PY=<指定解释器>
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MAIN_CODE=$(dirname "$SCRIPT_DIR")
PROJECT=$(dirname "$MAIN_CODE")
PY=${PY:-$PROJECT/.venv-d2l/bin/python}
CONFIG=${CONFIG:-$MAIN_CODE/configs/mab_sh6k.yaml}
LOG_DIR=$SCRIPT_DIR/logs

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

mkdir -p "$LOG_DIR"
# 日志按实验顺序编号:log-1.log、log-2.log …… 便于按次区分。
LAST=0
for f in "$LOG_DIR"/log-*.log; do
    [ -e "$f" ] || continue
    n=${f##*/log-}
    n=${n%.log}
    if [ "$n" -gt "$LAST" ]; then LAST=$n; fi
done
INDEX=$((LAST + 1))
LOG=$LOG_DIR/log-$INDEX.log

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

[ -x "$PY" ] || { echo "解释器不可执行: $PY" >&2; exit 2; }
[ -f "$CONFIG" ] || { echo "配置不存在: $CONFIG" >&2; exit 2; }

log "experiment #$INDEX host=$(hostname) job=${SLURM_JOB_ID:-none} gpu=${CUDA_VISIBLE_DEVICES:-unset}"
log "python=$PY"
log "config=$CONFIG"

# 配置与本地资源校验:不加载模型,先排除路径/字段漂移
log "step 1/3 validate-config"
PYTHONPATH=$MAIN_CODE/src "$PY" -m param_memory validate-config --config "$CONFIG" 2>&1 | tee -a "$LOG"

if [ "$CHECK_ONLY" -eq 1 ]; then
    log "check 通过(未运行实验)"
    exit 0
fi

# 没有 CUDA 就立刻失败,避免像上次那样跑几分钟才报 "没有可用 CUDA 设备"
if ! "$PY" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    log "当前进程没有可用 CUDA 设备;请在 GPU 分配内运行(bash)或直接 sbatch 本脚本"
    exit 3
fi
"$PY" -c 'import torch; print("gpu:", torch.cuda.get_device_name(0))' | tee -a "$LOG"

log "step 2/3 run-mab(100 题,耗时数小时)"
set +e
PYTHONPATH=$MAIN_CODE/src "$PY" -m param_memory run-mab --config "$CONFIG" 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}
set -e
log "run-mab exit=$STATUS"

log "step 3/3 汇总最新一次产物"
"$PY" - "$PROJECT/research/runs" <<'PY' | tee -a "$LOG"
import json, sys
from pathlib import Path

runs = Path(sys.argv[1])
dirs = sorted((d for d in runs.iterdir() if (d / "metrics.json").exists()), key=lambda d: d.stat().st_mtime)
if not dirs:
    print("未找到带 metrics.json 的产物目录")
    raise SystemExit(0)
run = dirs[-1]
m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
print(f"run_dir: {run}")
print(f"status: {manifest.get('status')}  failure: {manifest.get('failure')}")
print(f"questions: total={m['total_questions']} scored={m['scored_questions']} "
      f"failed={m['failed_questions']} not_attempted={m['not_attempted_questions']}")
print(f"score: substring_EM(分母固定)={m['failure_inclusive_score']:.4f}  "
      f"mean_scored={m['official_mean_scored']}")
print(f"cost: elapsed={m['elapsed_seconds']:.1f}s task_tokens_in={m.get('task_input_tokens')} "
      f"task_tokens_out={m.get('task_output_tokens')}")
print(f"maintenance: {m['maintenance_counts']} compiled_chunks={m['memory_usage']['compiled_chunks']}")
PY

exit "$STATUS"
