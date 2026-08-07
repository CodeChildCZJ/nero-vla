"""预热 + 诊断 WS infer 路径。dummy zeros obs, NERO 8维 state, 双cam 480x640x3。"""
import numpy as np, sys, time
from openpi_client import websocket_client_policy as wcp
PROMPT = "pick the pink sponge and place it in the blue bucket"
ports = [int(x) for x in sys.argv[1:]] or [9097, 9099]
for port in ports:
    try:
        t0 = time.time()
        c = wcp.WebsocketClientPolicy(host="127.0.0.1", port=port)
        obs = {
            "observation/image": np.zeros((480, 640, 3), np.uint8),
            "observation/wrist_image": np.zeros((480, 640, 3), np.uint8),
            "observation/state": np.zeros(8, np.float32),
            "prompt": PROMPT,
        }
        ch = np.asarray(c.infer(obs)["actions"])
        print(f"port {port}: WARM OK shape={ch.shape} {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"port {port}: ERR {type(e).__name__}: {e}")
