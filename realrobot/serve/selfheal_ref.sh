#!/bin/bash
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"
# 生产 ref 9229 cron 自愈: 每次 cron 调用检查, 不在就重起 (reaper 杀掉后自动补)
# crontab: */2 * * * * ${NERO_ROOT}/realrobot/selfheal_ref.sh
export PATH=$HOME/.local/bin:/usr/bin:/bin
# 已监听 → 啥也不做
ss -ltn 2>/dev/null | grep -q ":9229 " && exit 0
# 进程在(加载中) → 别重复起
pgrep -f "serve_policy.py.*--port 9229" >/dev/null && exit 0
# 真down → 起
cd ${NERO_ROOT}/realrobot
REF_CKPT_DIR=${NERO_CKPT}/b2_ref_KEEP/29999 SERVE_GPU=0 SERVE_MEM=0.25 \
  setsid bash ${NERO_ROOT}/realrobot/serve_b2_ref.sh 29999 9229 \
  >> ${NERO_ROOT}/realrobot/serve_b2_ref_29999.log 2>&1 < /dev/null &
echo "[$(date '+%F %T')] selfheal 重起 ref 9229" >> ${NERO_ROOT}/realrobot/selfheal_ref.log
exit 0
