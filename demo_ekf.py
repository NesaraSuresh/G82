import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.sim.sim_env import SimEnv
from src.perception.aruco_detector import MarkerDetector
from src.estimation.ekf import EKF

WHEEL_R_EST, WHEEL_SEP_EST = 0.0290, 0.3650   # ~3% off calibrated values, on purpose
MARKER_XY = (1.999, 0.0)

env = SimEnv("assets/robotis_tb3/scene_track_a.xml")
env.reset(0.0, 0.1, -0.05)
det = MarkerDetector(env.camera_intrinsics(), marker_len=0.2)

x0 = [0.0, 0.0, 0.0]                           # initial guess (truth is 0, 0.1, -0.05)
P0 = np.diag([0.1**2, 0.2**2, 0.1**2])
ekf = EKF(x0, P0, WHEEL_R_EST, WHEEL_SEP_EST, MARKER_XY)
dr = EKF(x0, P0, WHEEL_R_EST, WHEEL_SEP_EST, MARKER_XY)   # odometry only, never updated
l, r = env.encoders()
ekf.init_encoders(l, r)
dr.init_encoders(l, r)

truth, est, dead = [], [], []
printed = False
for k in range(600):                           # 12 s
    env.step(0.12, 0.25 * np.sin(2 * np.pi * k / 300))
    l, r = env.encoders()
    ekf.predict(l, r)
    dr.predict(l, r)
    if k % 5 == 0:                             # camera at 10 Hz
        z = det.measure(env.camera_image())
        if z is not None:
            if not printed:
                print("first measurement [range, bearing]:", z)
                print("expected from true pose:           ", ekf.h(env.true_pose()))
                printed = True
            ekf.update(z)
    truth.append(env.true_pose())
    est.append(ekf.x.copy())
    dead.append(dr.x.copy())

truth, est, dead = map(np.array, (truth, est, dead))
t = np.arange(len(truth)) * env.dt
err_ekf = np.linalg.norm(est[:, :2] - truth[:, :2], axis=1)
err_dr = np.linalg.norm(dead[:, :2] - truth[:, :2], axis=1)
print(f"final position error: EKF {err_ekf[-1]:.3f} m, odometry only {err_dr[-1]:.3f} m")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(truth[:, 0], truth[:, 1], "k", label="true")
ax[0].plot(dead[:, 0], dead[:, 1], "r--", label="odometry only")
ax[0].plot(est[:, 0], est[:, 1], "b", label="EKF")
ax[0].plot(*MARKER_XY, "g*", ms=14, label="marker")
ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)"); ax[0].axis("equal"); ax[0].legend()
ax[1].plot(t, err_dr, "r--", label="odometry only")
ax[1].plot(t, err_ekf, "b", label="EKF")
ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("position error (m)"); ax[1].legend()
plt.tight_layout()
plt.savefig("ekf_demo.png", dpi=120)
print("saved ekf_demo.png")
print("final estimate:", est[-1], " truth:", truth[-1])
print("EKF std [x, y, theta]:", np.sqrt(np.diag(ekf.P)))
