import numpy as np
from src.sim.sim_env import SimEnv

env = SimEnv("assets/robotis_tb3/scene_track_a.xml")

# 1) straight line: effective wheel radius
env.reset(0, 0, 0)
l0, r0 = env.encoders()
x0, y0, _ = env.true_pose()
for _ in range(200):
    env.step(0.10, 0.0)
l1, r1 = env.encoders()
x1, y1, _ = env.true_pose()
dist = np.hypot(x1 - x0, y1 - y0)
dphi = 0.5 * ((l1 - l0) + (r1 - r0))
r_eff = dist / dphi
print(f"effective wheel radius: {r_eff:.4f} m (script assumed 0.033)")

# 2) spin in place: effective wheel separation
env.reset(0, 0, 0)
l0, r0 = env.encoders()
for _ in range(150):
    env.step(0.0, 0.5)
l1, r1 = env.encoders()
_, _, yaw = env.true_pose()
sep_eff = r_eff * ((r1 - r0) - (l1 - l0)) / yaw
print(f"true rotation {yaw:.3f} rad, effective wheel separation: {sep_eff:.4f} m (script assumed 0.288)")
