import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.sim.sim_env import SimEnv
from src.perception.aruco_detector import MarkerDetector
from src.estimation.ekf import EKF
from src.control.lqr_dock import DockController

R_EST, SEP_EST = 0.0290, 0.3650     # slightly wrong on purpose
MARKER_XY = (1.999, 0.0)
GOAL_X = 1.45                        # robot origin stops here (marker stays in camera view)

env = SimEnv("assets/robotis_tb3/scene_track_a.xml")
start = (0.0, 0.3, -0.1)             # truth
env.reset(*start)
det = MarkerDetector(env.camera_intrinsics(), marker_len=0.2)

x0 = [start[0] + 0.05, start[1] - 0.05, start[2] + 0.10]   # imperfect initial guess
P0 = np.diag([0.1**2, 0.2**2, 0.15**2])
ekf = EKF(x0, P0, R_EST, SEP_EST, MARKER_XY)
ctrl = DockController(GOAL_X)
l, r = env.encoders()
ekf.init_encoders(l, r)

truth, est = [], []
n_det = n_cam = 0
for k in range(2500):                # up to 50 s
    v, w, done = ctrl.command(ekf.x)  # controller sees the ESTIMATE only
    if done:
        break
    env.step(v, w)
    l, r = env.encoders()
    ekf.predict(l, r)
    if k % 5 == 0:                    # camera at 10 Hz
        n_cam += 1
        z = det.measure(env.camera_image())
        if z is not None:
            ekf.update(z)
            n_det += 1
    truth.append(env.true_pose())
    est.append(ekf.x.copy())
else:
    print("TIMEOUT: did not reach the goal")

truth, est = np.array(truth), np.array(est)
t = np.arange(len(truth)) * env.dt
x, y, th = env.true_pose()
print(f"stopped after {len(truth) * env.dt:.1f} s")
print(f"true final pose: x={x:.3f} y={y:.3f} yaw={np.degrees(th):.1f} deg (goal x={GOAL_X}, y=0, yaw=0)")
print(f"docking error: along {x - GOAL_X:+.3f} m, lateral {y:+.3f} m, heading {np.degrees(th):+.1f} deg")
print("estimate error [x, y, yaw]:", np.round(ekf.x - np.array([x, y, th]), 3))
print(f"marker updates: {n_det} of {n_cam} camera frames")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(truth[:, 0], truth[:, 1], "k", label="true")
ax[0].plot(est[:, 0], est[:, 1], "b--", label="EKF estimate")
ax[0].plot(*MARKER_XY, "g*", ms=14, label="marker")
ax[0].plot(GOAL_X, 0, "rx", ms=10, label="goal")
ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)"); ax[0].axis("equal"); ax[0].legend()
ax[1].plot(t, truth[:, 1], "k", label="true"); ax[1].plot(t, est[:, 1], "b--", label="EKF")
ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("y (m)"); ax[1].legend()
ax[2].plot(t, np.degrees(truth[:, 2]), "k", label="true"); ax[2].plot(t, np.degrees(est[:, 2]), "b--", label="EKF")
ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("heading (deg)"); ax[2].legend()
plt.tight_layout()
try:
    plt.savefig("dock_demo.png", dpi=120)
    print("saved dock_demo.png")
except OSError:
    plt.savefig("dock_demo_new.png", dpi=120)
    print("dock_demo.png is open in a viewer, saved dock_demo_new.png instead")
