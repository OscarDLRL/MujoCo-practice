import numpy as np


class FootTrajectoryGenerator:
    def __init__(self, swing_height=0.05):
        self.swing_height = swing_height

    def swing(self, p0, pf, phase):
        """
        p0: posición inicial del pie
        pf: posición final del pie
        phase: 0 → 1

        devuelve posición del pie en swing
        """
        phase = np.clip(phase, 0.0, 1.0)

        # interpolación horizontal
        p = (1.0 - phase) * p0 + phase * pf

        # levantar el pie
        p[2] += self.swing_height * np.sin(np.pi * phase)

        return p