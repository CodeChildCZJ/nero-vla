# 上游依赖

本仓库**不重分发上游代码**。`third_party/` 在你 clone 之后是空的(只有本文件),需要时用:

```bash
bash tools/bootstrap_upstreams.sh            # 六个全拉
bash tools/bootstrap_upstreams.sh lerobot    # 只拉一个
```

脚本会把每个上游 clone 下来、checkout 到下表钉死的 commit、再把 `patches/` 里我们的改动打上去。

**只想复现榜单的话什么都不用拉** —— `bench/scripts/score/score.py` 只依赖 numpy。

> **为什么是脚本而不是 git submodule。** 我们对四个上游的改动都以普通 diff 存在 `patches/` 里,
> 并且逐份验证过能在钉死的 SHA 上干净应用。「clone 上游 + 打 patch」能精确重建我们用的那棵树,
> 不需要任何人长期维护四个 fork,上游 fork 被删或被 force-push 也不影响。
> 实测:对 `openvla-oft` 跑一遍 bootstrap,重建结果与我们原始工作树逐文件比对,
> 差异只有两个文件 —— 恰好是发布时把硬编码默认路径改成环境变量的那两个。

## 六个上游

| `third_party/` 下的目录 | 上游 | 我们改了什么 |
|---|---|---|
| `openpi-agilex` | [agilexrobotics/openpi-agilex](https://github.com/agilexrobotics/openpi-agilex) @ `432577c37c4f` | `src/openpi/training/config.py` **+287 行纯新增**(全部 `pi05_nero_*` 配置);新增 `src/openpi/policies/nero_policy.py`(127 行) |
| `openpi` | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) @ `c23745b5ad24` | **+435 行**:`training/config.py` 里 v1→v8 共 9 个 `pi05_nero_pick_pink_sponge*` 配置,外加 `scripts/train.py` 与 `serving/websocket_policy_server.py` 的改动。**`realrobot/train/train_v{2..8}.sh` 和 `serve_v*.sh` 引用的就是这些配置** —— 没有这个上游,真机线早期那一半跑不起来 |
| `Isaac-GR00T-n1d5` | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) @ `4af2b622892f` | `scripts/gr00t_finetune.py` 改 2 处(`save_total_limit=1` + `save_only_model=True`,磁盘所迫);新增 `setup_env.sh` |
| `openvla-oft` | [moojink/openvla-oft](https://github.com/moojink/openvla-oft) @ `e4287e9` | **改了 6 个上游文件 + 新增 2 个**,见下节 |
| `Isaac-GR00T` (N1.7) | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) @ `b9955401d50c` | **零改动** —— N1.7 的定制全在 `bench/configs/` 和 `bench/scripts/` |
| `lerobot` (ACT / SmolVLA) | [huggingface/lerobot](https://github.com/huggingface/lerobot) @ `f66e5128ecb2` | **零改动** —— 定制全在 `bench/scripts/` |

前四个有我们的改动(由 `patches/` 提供),后两个是原样的上游。

**两代 openpi 并存不是冗余。** 真机线跨了一次栈迁移:v1→v8 跑在 Physical-Intelligence 的 openpi 上,
b1/b2/ref 那一代迁到了 agilex 的 fork(它带 ALOHA/cobot_magic 的适配)。两边的配置互不通用,
所以两个仓库都得钉住,否则 `realrobot/` 里会有一半脚本引用不存在的配置名。

## 每个上游的源码树放在哪

所有六个上游一律落在 `third_party/<repo>` 下,脚本引用的也是这个路径。各栈的 venv 按惯例
建在各自的仓库里(`third_party/<repo>/.venv`),`env.example.sh` 里的 `NERO_LEROBOT_PY` 就是这么指的。
`third_party/` 在你跑 `tools/bootstrap_upstreams.sh` 之前是空的,所以任何需要跑训练/预测的脚本
在新 clone 的仓库里都会找不到路径 —— 这是预期行为,不是 bug。只跑 `score.py` 不受影响。

## openvla-oft 的改动面(比预想的大,这里如实列出)

最初以为只是加了一个 `finetune_nero.py`。实际反推 upstream diff 后发现改了 6 个上游文件:

**改动的 6 个文件 —— 全部是同一件事的两个侧面:**

| 文件 | 改了什么 | 为什么 |
|---|---|---|
| `prismatic/__init__.py` | eager re-export 包进 `try/except ImportError` | 上游这条 import 链会拉起 TensorFlow / RLDS / dlimp 全套预训练数据管线 |
| `prismatic/models/__init__.py` | 同上 | 同上 |
| `prismatic/vla/__init__.py` | 同上 | 同上 |
| `prismatic/vla/datasets/__init__.py` | 同上 | 同上 |
| `experiments/robot/openvla_utils.py` | `import tensorflow` 改为惰性;加 `from __future__ import annotations` | TF 只被三个图像 helper 用到,而 NERO 喂进去的帧已经是目标尺寸、不做 center crop |
| `prismatic/vla/constants.py` | 新增 `NERO_CONSTANTS` | 8 维绝对关节目标 / chunk 8 / 30fps;用 `BOUNDS` 而非 `BOUNDS_Q99`,理由跟上游 ALOHA 一样 —— 动作空间是**绝对**关节角,截断离群值会让某些可达角度变得不可表示 |

**新增的 2 个文件:** `vla-scripts/finetune_nero.py`(1188 行)、`prismatic/vla/datasets/nero_dataset.py`(261 行,map-style torch Dataset,绕开 RLDS)。

前 5 项本质上是**一个改动**: 让 `prismatic` 在没装 TF/RLDS 的环境里也能 import。我们用 map-style torch Dataset 替代了 RLDS 管线,所以那套依赖没装。**在装了完整上游依赖的环境里,这些改动的行为与上游完全一致** —— try 块会正常成功。

## patches/

`patches/` 目录保存了这些改动的原始 diff 和新增文件,供审计,或在别的上游版本上重放:

```
patches/
├─ openpi.patch                        # v1→v8 的 9 个配置 + train.py + websocket server
├─ openpi-agilex.patch                 # config.py 的 +287 行
├─ openpi-agilex.nero_policy.py        # 新增文件(untracked,git diff 抓不到)
├─ Isaac-GR00T-n1d5.patch
├─ Isaac-GR00T-n1d5.setup_env.sh       # 新增文件
├─ openvla-oft.patch                   # 6 个上游文件的改动
├─ openvla-oft.finetune_nero.py        # 新增文件
└─ openvla-oft.nero_dataset.py         # 新增文件
```

四份 `.patch` 都经过验证:`openvla-oft.patch` 在上游 `e4287e9` 上 `git apply --check` 干净通过;
另外三份(`openpi`、`openpi-agilex`、`Isaac-GR00T-n1d5`)用 `git apply --check --reverse` 验证过,
确实忠实捕获了我们工作树里的改动 —— 不是"生成了一个文件"就算数。

**一个诚实的保留:** `openvla-oft` 的本地副本是被剥掉 `.git` 后 vendored 进来的,所以无法确知它当初 fork 自哪个 commit。上表里的 `e4287e9` 是 2026-08-07 反推 diff 时上游的 HEAD。之所以认为这个基准是对的:反推出的每一个 hunk 都带着我们自己的 NERO 注释,没有任何一处看起来像无关的上游变更 —— 但这是推断,不是记录。
