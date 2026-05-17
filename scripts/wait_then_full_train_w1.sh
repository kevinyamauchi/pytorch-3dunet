#!/usr/bin/env bash
set -euo pipefail
ROOT_PID="${1:?usage: wait_then_full_train_w1.sh PARENT_PID}"
REPO="/local0/home/lfranz/code/pytorch-3dunet"
VENV="${REPO}/.venv/bin/python"
MON="${REPO}/scripts/stimcycle_training_memory.py"

echo "$(date -Is) waiting for parent pid ${ROOT_PID} to exit..."
while kill -0 "${ROOT_PID}" 2>/dev/null; do
  sleep 30
done
echo "$(date -Is) parent exited; starting w=1 full training with monitor..."

exec /usr/bin/env -i \
  HOME="${HOME:-/local0/home/lfranz}" \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  PATH="/local0/home/lfranz/.local/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin" \
  "${VENV}" "${MON}" run \
  --full-training \
  --batch-size 15 \
  --num-workers 1 \
  --run-name full_default_w1_seed0 \
  --sample-interval 10 \
  --timeout-seconds 0
