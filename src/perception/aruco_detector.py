import cv2
import numpy as np


class MarkerDetector:
    """Detects an ArUco marker and returns [range, bearing] in the camera's
    ground plane. Bearing is positive to the left (counter-clockwise)."""

    def __init__(self, K, marker_len=0.2, marker_id=0):
        self.K = np.asarray(K, dtype=np.float64)
        self.marker_id = marker_id
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        h = marker_len / 2
        self.obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]],
                            dtype=np.float32)

    def measure(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or self.marker_id not in ids.ravel():
            return None
        i = list(ids.ravel()).index(self.marker_id)
        pts = corners[i].reshape(4, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(self.obj, pts, self.K, None,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None
        x, _, z = tvec.ravel()      # OpenCV camera frame: x right, y down, z forward
        return np.array([np.hypot(x, z), np.arctan2(-x, z)])
