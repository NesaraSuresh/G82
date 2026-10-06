import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class EKF:
    def __init__(self, x0, P0, wheel_r, wheel_sep, marker_xy,
                 cam_offset=0.09, odom_k=2e-3, R_meas=None):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.r, self.L = wheel_r, wheel_sep
        self.m = np.array(marker_xy, dtype=float)
        self.a = cam_offset                 # camera is this far ahead of the robot origin
        self.k = odom_k                     # odometry variance per metre travelled
        self.R = np.diag([0.05**2, np.deg2rad(2.0)**2]) if R_meas is None else R_meas
        self.prev = None

    def init_encoders(self, l, r):
        self.prev = (l, r)

    def predict(self, l, r):
        dl = (l - self.prev[0]) * self.r
        dr = (r - self.prev[1]) * self.r
        self.prev = (l, r)
        ds, dth = 0.5 * (dr + dl), (dr - dl) / self.L
        x, y, th = self.x
        thm = th + 0.5 * dth
        c, s = np.cos(thm), np.sin(thm)
        self.x = np.array([x + ds * c, y + ds * s, wrap(th + dth)])

        F = np.array([[1, 0, -ds * s], [0, 1, ds * c], [0, 0, 1]])
        Gu = np.array([[0.5 * c - ds / (2 * self.L) * s, 0.5 * c + ds / (2 * self.L) * s],
                       [0.5 * s + ds / (2 * self.L) * c, 0.5 * s - ds / (2 * self.L) * c],
                       [1 / self.L, -1 / self.L]])
        Su = np.diag([self.k * abs(dr), self.k * abs(dl)])
        Q = Gu @ Su @ Gu.T + np.diag([1e-8, 1e-8, 1e-8])
        self.P = F @ self.P @ F.T + Q

    def h(self, state):
        """Expected [range, bearing] to the marker from a given state."""
        x, y, th = state
        dx = self.m[0] - (x + self.a * np.cos(th))
        dy = self.m[1] - (y + self.a * np.sin(th))
        return np.array([np.hypot(dx, dy), wrap(np.arctan2(dy, dx) - th)])

    def update(self, z):
        x, y, th = self.x
        a = self.a
        dx = self.m[0] - (x + a * np.cos(th))
        dy = self.m[1] - (y + a * np.sin(th))
        q = dx * dx + dy * dy
        r = np.sqrt(q)
        H = np.array([
            [-dx / r, -dy / r, a * (dx * np.sin(th) - dy * np.cos(th)) / r],
            [dy / q, -dx / q, -a * (dy * np.sin(th) + dx * np.cos(th)) / q - 1.0]])
        nu = z - self.h(self.x)
        nu[1] = wrap(nu[1])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        self.x[2] = wrap(self.x[2])
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return nu
