import torch

# ---THRUSTER DATA FOR CSAD (Converted to Torch)---
class ThrusterData:
    """ Container for thruster-related parameters as PyTorch tensors. """
    propeller_diameter = torch.tensor(0.03, dtype=torch.float32)  # [m]
    n_dot_max = torch.tensor(5.0, dtype=torch.float32)            # [1/s^2]
    alpha_dot_max = torch.tensor(2.0, dtype=torch.float32)        # [1/s]
    thrust_max = torch.tensor(1.5, dtype=torch.float32)           # [N]
    thruster_min = torch.tensor(-0.85, dtype=torch.float32)       # [N]

    # Thruster Positions
    lx = torch.tensor([1.0678, 0.9344, 0.9344, -1.1644, -0.9911, -0.9911], dtype=torch.float32)
    ly = torch.tensor([0.0, 0.11, -0.11, 0.0, -0.1644, 0.1644], dtype=torch.float32)

    # Thrust Coefficients
    K = torch.tensor([1.491, 1.491, 1.491, 1.491, 1.491, 1.491], dtype=torch.float32)  # Thrust coefficient

# ----------------------------------------------------------------
# PyTorch-Based Thruster Dynamics for Differentiable Simulation
# ----------------------------------------------------------------
class ThrusterDynamics(torch.nn.Module):
    def __init__(self):
        """
        Thruster dynamics class using PyTorch.
        Implements thrust computation, rate limiting, and saturation constraints.
        """
        super().__init__()
        self.register_buffer("_K", torch.diag(ThrusterData.K))  # Diagonal thrust coefficient matrix
        self.register_buffer("_lx", ThrusterData.lx)  # x-positions
        self.register_buffer("_ly", ThrusterData.ly)  # y-positions
        self._n_thrusters = len(ThrusterData.lx)

    def saturate(self, signal, min_val, max_val):
        """
        Saturates a given signal with both an upper and lower bound.

        Parameters:
        - signal (tensor): Input signal.
        - min_val (tensor or float): Lower bound.
        - max_val (tensor or float): Upper bound.

        Returns:
        - (tensor): Saturated signal.
        """
        return torch.clamp(signal, min_val, max_val)

    def limit_rate(self, signal_curr, signal_prev, max_rate, dt):
        """
        Limits the rate of change of the signal with respect to a maximum rate.

        Parameters:
        - signal_curr (tensor): Current signal.
        - signal_prev (tensor): Previous signal.
        - max_rate (float): Maximum rate of change.
        - dt (float): Time step.

        Returns:
        - (tensor): Rate-limited signal.
        """
        delta = signal_curr - signal_prev
        limit = max_rate * dt
        return signal_prev + torch.clamp(delta, -limit, limit)

    def propeller_revolution(self, u):
        """
        Computes propeller revolution numbers from control inputs.

        Parameters:
        - u (tensor): Control signal.

        Returns:
        - (tensor): Propeller revolution numbers.
        """
        return torch.sign(u) * torch.sqrt(torch.abs(u) + 1e-8)  # Added small epsilon for stability

    def control_input(self, n):
        """
        Computes the control input from the propeller revolution numbers.

        Parameters:
        - n (tensor): Propeller revolution numbers.

        Returns:
        - (tensor): Control signal.
        """
        return n * torch.abs(n)

    def actuator_loads(self, u):
        """
        Computes load on each actuator from the control inputs.

        Parameters:
        - u (tensor): Control signal.

        Returns:
        - (tensor): Actuator loads.
        """
        return self._K @ u  # Multiply thrust coefficients with control inputs

    def thruster_configuration(self, alpha):
        """
        Computes the thruster configuration matrix.

        Parameters:
        - alpha (tensor): Azimuth angles.

        Returns:
        - (tensor): Thrust configuration matrix.
        """
        return torch.stack([
            torch.cos(alpha),
            torch.sin(alpha),
            self._lx * torch.sin(alpha) - self._ly * torch.cos(alpha)
        ])

    def get_tau(self, u, alpha):
        """
        Computes the resulting forces and moments in surge, sway, and yaw.

        Parameters:
        - u (tensor): Control input (thrust per thruster).
        - alpha (tensor): Thruster angles.

        Returns:
        - (tensor): Resulting force/moment vector (3D: surge, sway, yaw).
        """
        return self.thruster_configuration(alpha) @ self.actuator_loads(u)
