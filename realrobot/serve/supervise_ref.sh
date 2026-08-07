#!/bin/bash
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"
# 生产 ref 9229 看门狗: serve 被 reaper 杀掉后立即自动重起 (不依赖外部监控)
# 外层 while 进程 cmdline=bash 不含 serve_policy.py, 不会被按名 pkill 误杀
# 起法: setsid bash ${NERO_ROOT}/realrobot/supervise_ref.sh >/dev/null 2>&1 </dev/null & disown
SUPLOG=${NERO_ROOT}/realrobot/supervise_ref.log
echo "[$(date '+%F %T')] supervisor 启动" >> "$SUPLOG"
while true; do
  REF_CKPT_DIR=${NERO_CKPT}/b2_ref_KEEP/29999 SERVE_GPU=0 SERVE_MEM=0.25 \
    bash ${NERO_ROOT}/realrobot/serve_b2_ref.sh 29999 9229 \
    >> ${NERO_ROOT}/realrobot/serve_b2_ref_29999.log 2>&1
  code=$?
  echo "[$(date '+%F %T')] ref serve 退出(code $code), 3s 后重起" >> "$SUPLOG"
  sleep 3
done
