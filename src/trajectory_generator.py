import numpy as np

class WaypointTrajectory:
    def __init__(self, waypoints, speed=0.3, dt=0.02):
        self.waypoints = np.array(waypoints, dtype=float)
        self.speed = speed
        self.dt = dt
        self.samples = self._build_trajectory()

    def _build_trajectory(self):
        traj = []
        for i in range(len(self.waypoints) - 1):
            p0 = self.waypoints[i]
            p1 = self.waypoints[i + 1]
            delta = p1 - p0
            dist = np.linalg.norm(delta[:2])
            steps = max(2, int(dist / (self.speed * self.dt)))
            for a in np.linspace(0.0, 1.0, steps, endpoint=False):
                p = (1 - a) * p0 + a * p1
                yaw = np.arctan2(delta[1], delta[0])
                vx = self.speed * np.cos(yaw)
                vy = self.speed * np.sin(yaw)
                traj.append({
                    "pos": np.array([p[0], p[1], p[2] if len(p) > 2 else 0.0]),
                    "vel": np.array([vx, vy, 0.0]),
                    "yaw": yaw,
                    "yaw_rate": 0.0
                })
        traj.append(traj[-1])
        return traj

    def get_reference(self, k):
        return self.samples[min(k, len(self.samples) - 1)]

    def length(self):
        return len(self.samples)