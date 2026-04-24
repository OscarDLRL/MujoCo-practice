import numpy as np

class TrotGaitPlanner:
    def __init__(self, step_period=2.0):
        self.step_period = step_period
        self.swing_order = [2, 0, 3, 1]  # RL, FL, RR, FR

    def get_phase(self, t):
        return (t % self.step_period) / self.step_period

    def get_active_swing_leg(self, t):
        phase = self.get_phase(t)
        section = min(int(phase * 4.0), 3)
        return self.swing_order[section], section

    def get_contact_pattern(self, t):
        contact = np.ones(4, dtype=bool)
        leg_id, _ = self.get_active_swing_leg(t)
        contact[leg_id] = False
        return contact

    def get_swing_phase(self, t, leg_id):
        phase = self.get_phase(t)
        active_leg, section = self.get_active_swing_leg(t)
        if leg_id != active_leg:
            return 0.0
        local_phase = phase * 4.0 - section
        return float(np.clip(local_phase, 0.0, 1.0))