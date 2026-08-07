"""真机回环验收 (windows 协议): 喂已知 deg+mm state, 验 server 端转换层透明。
通过判据: 返回 chunk 关节量级=deg(几十~百, 非rad±3); 夹爪=mm(0~76, 非[0,1])。
用法: python test_ref_loopback.py <port>
"""
import sys
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

port = int(sys.argv[1]) if len(sys.argv) > 1 else 9221

# 已知 deg state: j1=-2.86(windows指定), 其余取 grasp 构型量级 (j4~70,j7~38), 夹爪 40mm(半开)
state_deg = np.array([-2.86, 10.0, -5.0, 70.0, 0.0, 20.0, 38.0, 40.0], dtype=np.float32)
img = np.zeros((480, 640, 3), dtype=np.uint8)  # 黑图: 量级判据与图像无关
obs = {
    "observation/image": img,
    "observation/wrist_image": img,
    "observation/state": state_deg,
    "prompt": "pick the pink sponge and place it in the blue bucket",
}

c = WebsocketClientPolicy(host="127.0.0.1", port=port)
out = c.infer(obs)
a = np.asarray(out["actions"], dtype=np.float32)   # 期望 [10, 8]

jmin, jmax = float(a[:, :7].min()), float(a[:, :7].max())
gmin, gmax = float(a[:, 7].min()), float(a[:, 7].max())
jabs = float(np.abs(a[:, :7]).max())

print(f"== port {port} 回环结果 ==")
print(f"chunk shape: {a.shape}")
print(f"关节 range: [{jmin:.2f}, {jmax:.2f}] deg  | |max|={jabs:.2f}")
print(f"夹爪 range: [{gmin:.2f}, {gmax:.2f}]")
print(f"chunk[0] = {np.round(a[0], 2).tolist()}")
joint_ok = jabs > 5.0          # deg 量级 (rad 关节 |max|≈≤3.5); >5 说明已*57.3
grip_ok = gmax > 2.0           # mm 量级 (>2 说明不是[0,1]); 上限~76
print(f"关节量级=deg: {'✅' if joint_ok else '❌ 仍是rad!'}  夹爪量级=mm: {'✅' if grip_ok else '❌ 仍是[0,1]!'}")
print("LOOPBACK_PASS" if (joint_ok and grip_ok) else "LOOPBACK_FAIL")
