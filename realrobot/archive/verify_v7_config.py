#!/usr/bin/env python3
"""验证 v7 config 写对了 (不碰数据集, 只建 TrainConfig 对象)。"""
from openpi.training import config as _config
c = _config.get_config("pi05_nero_pick_pink_sponge_v7")
print("name           =", c.name)
print("pi05           =", c.model.pi05)
print("action_horizon =", c.model.action_horizon)
print("action_dim     =", c.model.action_dim)
print("mask_gripper   =", c.data.mask_gripper_state, "  (应 False = 去mask)")
print("ema_decay      =", c.ema_decay, "  (应 0.99)")
print("num_train_steps=", c.num_train_steps)
print("batch_size     =", c.batch_size)
print("save_interval  =", c.save_interval, " keep_period =", c.keep_period)
print("repo_id        =", c.data.repo_id)
print("weight_loader  =", type(c.weight_loader).__name__)
print("\n[OK] v7 config 加载成功, 参数正确即就绪 (等数据建symlink+norm_stats)")
