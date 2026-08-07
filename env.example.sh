#!/bin/bash
# 复制为 env.sh 并按你的机器填写,然后 `source env.sh`。
#
# 注意:仅“重新训练 / 重跑预测”需要这些变量。
# 只想复现榜单的话什么都不用设 —— 直接 python bench/scripts/score/score.py。

# 本仓库 clone 到哪里
export NERO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LeRobot 本地数据集的【根目录】。真机线(realrobot/)引用了 v2_clean / v3 / v7 / v8 / b1 等
# 多份数据集,靠这个根目录 + 各自的目录名定位,例如 $NERO_DATA_ROOT/pick_pink_sponge_v7。
export NERO_DATA_ROOT="$HOME/.cache/huggingface/lerobot/local"

# 离线横评(bench/)用的那【一份】数据集 = b2。数据集本身不公开,格式规范见 docs/DATA.md。
export NERO_DATA="$NERO_DATA_ROOT/pick_pink_sponge_b2"

# checkpoint 输出根目录
export NERO_CKPT="$NERO_ROOT/checkpoints"

# wandb(可选;不设则各训练脚本以离线模式运行)
export WANDB_ENTITY="your-wandb-entity"
export WANDB_PROJECT="nero_backbone_bench"

# ---------------------------------------------------------------------------
# 下面这些都有合理默认值,通常不用设。列在这里是为了让你知道它们存在,
# 以及在你的目录结构跟默认不一样时该覆盖哪一个。
# ---------------------------------------------------------------------------

# 由 b2 派生出来的两份工作副本(脚本会自己生成)。默认是在 $NERO_DATA 后面加后缀。
export NERO_DATA_BENCH="${NERO_DATA}_bench"      # LeRobot v3.0 转换后的副本(ACT / SmolVLA 用)
export NERO_DATA_TRAIN="${NERO_DATA}_train"      # 只含 111 个 train episode 的子集(openpi 用)
export NERO_DATA_V30="$NERO_DATA_BENCH"          # 同上,v3.0 副本的另一个入口名

# LeRobot submodule 里那个 venv 的 python(ACT / SmolVLA 的训练与预测走它)
export NERO_LEROBOT_PY="$NERO_ROOT/third_party/lerobot/.venv/bin/python"

# GR00T N1.7 的 VLM 底模本地快照。上游默认那个是 gated 的;
# 指向一个未 gate 的等价 checkpoint 即可。
export NERO_BACKBONE_MODEL="$NERO_CKPT/Qwen3-VL-2B-Instruct"

# --- 只有跨机训练才需要 ---------------------------------------------------
# 我们把 pi0 和 GR00T N1.7 放在第二台机器上训,产物再拉回打分机。
# 单机跑的话这两个完全不用设,相关脚本你也不会用到。
export NERO_REMOTE_HOST="user@your-second-training-host"
export NERO_REMOTE_SSH_OPTS="-o BatchMode=yes"    # 例如非默认端口就写 "-p 2222 -o BatchMode=yes"

# 硬件断言:把一个 14 小时的训练误扔到同事的卡上是很贵的错误,
# 所以 n17_go.sh 会先断言自己确实跑在你指定的这台机器上。
export NERO_EXPECTED_HOST="$(hostname)"

# 冒烟模式:只在可见设备上把 model+processor+dataset 建起来就退出,不真训。
export NERO_DRYRUN=0
