import numpy as np
import mujoco

WHEEL_R = 0.0282      # effective (calibrated) radius
WHEEL_SEP = 0.3558    # effective (calibrated) separation


class SimEnv:
    """MuJoCo wrapper. The controller must only use step(), encoders(),
    camera_image(), camera_intrinsics() and lidar(). true_pose() is for
    plotting and error metrics only."""

    def __init__(self, scene_path, height=480, width=640, ctrl_dt=0.02, seed=0):
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.H, self.W = height, width
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.rng = np.random.default_rng(seed)
        self.n_sub = max(1, int(round(ctrl_dt / self.model.opt.timestep)))
        self.dt = self.n_sub * self.model.opt.timestep

        self.base_id = self.model.body("base").id
        self.lidar_site = self.model.site("lidar").id
        self.cam_name = "front_cam"
        self.qadr_l = self.model.jnt_qposadr[self.model.joint("wheel_left").id]
        self.qadr_r = self.model.jnt_qposadr[self.model.joint("wheel_right").id]
        self.enc_noise = 0.0    # std of encoder angle noise (rad)

    def reset(self, x=0.0, y=0.0, yaw=0.0):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [x, y, 0.0]
        self.data.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)

    def step(self, v, w):
        """Command forward speed v (m/s) and turn rate w (rad/s)."""
        wl = (v - w * WHEEL_SEP / 2) / WHEEL_R
        wr = (v + w * WHEEL_SEP / 2) / WHEEL_R
        self.data.ctrl[0] = wl   # actuator order: wheel_left, wheel_right
        self.data.ctrl[1] = wr
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)

    def encoders(self):
        """Cumulative wheel angles (rad), left and right, with optional noise."""
        l = self.data.qpos[self.qadr_l] + self.rng.normal(0, self.enc_noise)
        r = self.data.qpos[self.qadr_r] + self.rng.normal(0, self.enc_noise)
        return l, r

    def camera_image(self):
        self.renderer.update_scene(self.data, camera=self.cam_name)
        return self.renderer.render().copy()   # RGB, H x W x 3

    def camera_intrinsics(self):
        fovy = np.deg2rad(self.model.cam_fovy[self.model.camera(self.cam_name).id])
        f = (self.H / 2) / np.tan(fovy / 2)
        return np.array([[f, 0, self.W / 2], [0, f, self.H / 2], [0, 0, 1.0]])

    def lidar(self, n=360, max_range=3.5, sigma=0.01):
        """2D scan around the robot. Misses return np.inf."""
        origin = self.data.site_xpos[self.lidar_site].copy()
        R = self.data.site_xmat[self.lidar_site].reshape(3, 3)
        groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)  # group 0 only
        geomid = np.zeros(1, dtype=np.int32)
        ranges = np.full(n, np.inf)
        for i, a in enumerate(np.linspace(0, 2 * np.pi, n, endpoint=False)):
            d = R @ np.array([np.cos(a), np.sin(a), 0.0])
            dist = mujoco.mj_ray(self.model, self.data, origin, d, groups, 1,
                                 self.base_id, geomid)
            if 0 < dist <= max_range:
                ranges[i] = dist + self.rng.normal(0, sigma)
        return ranges

    def true_pose(self):
        """Ground truth (x, y, yaw). Do NOT feed this to the controller."""
        x, y = self.data.qpos[0:2]
        w, qx, qy, qz = self.data.qpos[3:7]
        yaw = np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        return x, y, yaw
