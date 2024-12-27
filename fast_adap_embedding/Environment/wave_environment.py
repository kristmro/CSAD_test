import numpy as np
from MCSimPython.simulator.csad import CSAD_DP_6DOF
from MCSimPython.waves.wave_loads import WaveLoad
from MCSimPython.waves.wave_spectra import JONSWAP

from MCSimPython.utils import three2sixDOF

class WaveEnvironment:
    """Environment for simulating vessel dynamics under varying wave conditions."""
    def __init__(self, dt=0.01):
        self.dt = dt
        self.vessel = CSAD_DP_6DOF(dt)
        self.waveload = None

    def set_wave_conditions(self, hs, tp, wave_dir_deg):
        """Set wave conditions for the environment."""
        wp = 2 * np.pi / tp
        N_w = 25  # Number of wave components
        wmin, wmax = wp / 2, 2.5 * wp
        dw = (wmax - wmin) / N_w
        w = np.linspace(wmin, wmax, N_w)

        jonswap = JONSWAP(w)
        freq, spec = jonswap(hs, tp, gamma=1.8)

        wave_amps = np.sqrt(2 * spec * dw)
        eps = np.random.uniform(0, 2 * np.pi, size=N_w)
        wave_dir = np.ones(N_w) * np.deg2rad(wave_dir_deg)

        self.waveload = WaveLoad(wave_amps, w, eps, wave_dir, config_file=self.vessel._config_file)

    def step(self, action=None):
        """Simulate a single time step."""
        tau_wf = self.waveload.first_order_loads(0, self.vessel.get_eta())
        tau_sv = self.waveload.second_order_loads(0, self.vessel.get_eta()[-1])
        tau_w = tau_wf + tau_sv  # 6DOF wave forces

        # Ensure the action is converted to 6DOF if provided
        if action is not None:
            # Explicitly expand action to 6DOF
            action_6dof = np.zeros(6)
            action_6dof[:3] = action[:3]  # Fill surge, sway, yaw
            self.vessel.integrate(0, 0, action_6dof + tau_w)
        else:
            self.vessel.integrate(0, 0, tau_w)

        return self.vessel.get_eta()

    def reset(self):
        """Reset the vessel and environment."""
        self.vessel = CSAD_DP_6DOF(self.dt)

    def get_state(self):
        """Get the current state of the vessel."""
        return self.vessel.get_eta()
