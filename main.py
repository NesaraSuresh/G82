import cv2
from src.sim.sim_env import SimEnv

env = SimEnv("assets/robotis_tb3/scene_track_a.xml")
env.reset(0.0, 0.0, 0.0)

for _ in range(150):          # 3 s at 50 Hz
    env.step(0.15, 0.0)       # drive straight

x, y, yaw = env.true_pose()
print(f"pose after 3 s: x={x:.3f} y={y:.3f} yaw={yaw:.3f}")
print("lidar range straight ahead:", env.lidar()[0])

img = env.camera_image()
cv2.imwrite("camera_test.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
corners, ids, _ = detector.detectMarkers(img)
print("marker ids detected:", None if ids is None else ids.ravel())
