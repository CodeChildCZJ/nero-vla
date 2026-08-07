#!/usr/bin/env python3
"""查 v6 server(9095) 实际返回的 action shape — 钉死 model 原生 horizon 真值。
client 契约: observation/image+wrist_image 224x224x3 uint8, state 32维, prompt。"""
import numpy as np
from openpi_client import websocket_client_policy as wcp

c = wcp.WebsocketClientPolicy(host="127.0.0.1", port=9095)
print("[meta]", c.get_server_metadata())
obs = {
    "observation/image": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
    "observation/wrist_image": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
    "observation/state": np.zeros(32, dtype=np.float32),
    "prompt": "pick the pink sponge and place it in the blue bucket",
}
out = c.infer(obs)
act = np.asarray(out["actions"])
print("[ACTIONS] shape =", act.shape, "dtype =", act.dtype)
print("[判决] model 原生 horizon =", act.shape[0], " action_dim =", act.shape[1] if act.ndim > 1 else "?")
print("[grip列(idx7)前几步] =", np.round(act[:, 7], 1).tolist() if act.ndim > 1 else "n/a")
