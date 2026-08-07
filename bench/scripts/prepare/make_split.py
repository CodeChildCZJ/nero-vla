import json, numpy as np, pathlib
# 仓库自包含:从本文件位置推导,不依赖任何环境变量或绝对路径。
# bench/scripts/prepare/make_split.py -> parents[2] == bench/
BENCH = pathlib.Path(__file__).resolve().parents[2]
N = 131                      # b2 total episodes
N_VAL = 20                   # ~15% held-out
SEED = 42
rng = np.random.default_rng(SEED)
perm = rng.permutation(N)
val = sorted(int(x) for x in perm[:N_VAL])
train = sorted(int(x) for x in perm[N_VAL:])
out = {
    "dataset": "local/pick_pink_sponge_b2",
    "total_episodes": N, "fps": 30,
    "n_train": len(train), "n_val": len(val),
    "seed": SEED,
    "val_episodes": val,
    "train_episodes": train,
    "note": "episode-level held-out; identical for ALL backbones; offline open-loop action-error on val anchors",
}
p = BENCH / "data" / "split.json"
p.write_text(json.dumps(out, indent=2))
print("wrote", p)
print("val_episodes:", val)
print("n_train", len(train), "n_val", len(val))
