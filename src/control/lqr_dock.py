import numpy as np
from scipy.linalg import solve_continuous_are

from src.estimation.ekf import wrap


class DockController:
    """Drives along the dock axis (y = 0, heading = 0) and stops at goal_x.
    Uses ONLY the estimated state, never ground truth.

    Error model about the dock axis, with forward speed v:
        d(e_y)/dt = v * e_theta,   d(e_theta)/dt = w
    The LQR gain is recomputed for the current v (gain scheduling)."""

    def __init__(self, goal_x, v_max=0.12, w_max=0.5, q=(10.0, 1.0), r=1.0, tol=0.02):
        self.goal_x, self.v_max, self.w_max, self.tol = goal_x, v_max, w_max, tol
        self.Q = np.diag(q)
        self.R = np.array([[r]])

    def gain(self, v):
        A = np.array([[0.0, v], [0.0, 0.0]])
        B = np.array([[0.0], [1.0]])
        P = solve_continuous_are(A, B, self.Q, self.R)
        return (B.T @ P / self.R[0, 0]).ravel()

    def command(self, x_est):
        x, y, th = x_est
        dist = self.goal_x - x
        if dist < self.tol:
            return 0.0, 0.0, True
        v = float(np.clip(0.5 * dist, 0.04, self.v_max))   # slow down near the goal
        k = self.gain(v)
        w = -(k[0] * y + k[1] * wrap(th))
        return v, float(np.clip(w, -self.w_max, self.w_max)), False
