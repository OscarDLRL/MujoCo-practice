import numpy as np


class FootTrajectoryGenerator:
    def __init__(self, swing_height=0.01):
        self.swing_height = swing_height

    def swing(self, p0, pf, phase):
        phase = np.clip(phase, 0.0, 1.0)

        p0 = np.asarray(p0, dtype=float)
        pf = np.asarray(pf, dtype=float)

        # Interpolación horizontal
        p = ((1.0 - phase) * p0 + phase * pf).copy()

        # Levantamiento suave
        p[2] += self.swing_height * np.sin(np.pi * phase)

        return p