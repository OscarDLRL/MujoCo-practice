import numpy as np


class TrotGaitPlanner:
    """
    Generador de marcha tipo trot.

    Orden de patas:
    0 = FL  front left
    1 = FR  front right
    2 = RL  rear left
    3 = RR  rear right
    """

    def __init__(self, step_period=0.6, duty_factor=0.6, swing_height=0.05):
        self.step_period = step_period
        self.duty_factor = duty_factor
        self.swing_height = swing_height

        # Fases para trot:
        # FL y RR juntas
        # FR y RL juntas
        self.phase_offsets = np.array([0.0, 0.5, 0.5, 0.0])

    def get_phase(self, t):
        """
        Fase global entre 0 y 1.
        """
        return (t % self.step_period) / self.step_period

    def get_leg_phase(self, t, leg_id):
        """
        Fase individual de cada pata entre 0 y 1.
        """
        phase = self.get_phase(t)
        return (phase + self.phase_offsets[leg_id]) % 1.0

    def is_leg_in_contact(self, t, leg_id):
        """
        True = pata en apoyo
        False = pata en swing
        """
        leg_phase = self.get_leg_phase(t, leg_id)
        return leg_phase < self.duty_factor

    def get_contact_pattern(self, t):
        """
        Devuelve contactos en orden:
        [FL, FR, RL, RR]
        """
        return np.array([
            self.is_leg_in_contact(t, 0),
            self.is_leg_in_contact(t, 1),
            self.is_leg_in_contact(t, 2),
            self.is_leg_in_contact(t, 3),
        ], dtype=bool)

    def get_swing_phase(self, t, leg_id):
        """
        Si la pata está en swing, devuelve fase de swing 0 a 1.
        Si está en apoyo, devuelve 0.
        """
        leg_phase = self.get_leg_phase(t, leg_id)

        if leg_phase < self.duty_factor:
            return 0.0

        return (leg_phase - self.duty_factor) / (1.0 - self.duty_factor)

    def get_swing_height(self, t, leg_id):
        """
        Altura vertical de la pata durante swing.
        Usa una curva senoidal:
        empieza en 0, sube, y termina en 0.
        """
        swing_phase = self.get_swing_phase(t, leg_id)

        if swing_phase <= 0.0:
            return 0.0

        return self.swing_height * np.sin(np.pi * swing_phase)