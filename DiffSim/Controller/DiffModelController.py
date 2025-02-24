import torch
import torch.nn as nn
from DiffSim.Simulator.DiffCsad import DiffCSAD_6DOF
from DiffSim.DiffUtils import six2threeDOF


class DiffModelController(nn.Module):
    """
    A simple model-based controller for the ship using the full dynamic model.
    
    Implements a PD control law with model-based compensation.
    """
    def __init__(self, dt):
        super().__init__()
        self.dt = dt

        # Load vessel dynamics (M and D matrices)
        self.simulator = DiffCSAD_6DOF(dt)
        self.M_inv = six2threeDOF(self.simulator._Minv)
        self.D = six2threeDOF(self.simulator._D)
        

        # PD Gains (adjustable)
        self.Kp = torch.tensor([5.0, 10.0, 1.5], dtype=torch.float32)  # Proportional gains
        self.Kd = torch.tensor([2.0, 2.0, 0.8], dtype=torch.float32)  # Derivative gains

    def compute_control(self, state, eta_d, nu_d, eta_d_ddot):
        """
        Compute control input using PD control + model-based compensation.

        Parameters:
        - state (dict): Current state of the vessel
        - eta_d (Tensor): Desired position [x, y, psi]
        - nu_d (Tensor): Desired velocity [u, v, r]
        - eta_d_ddot (Tensor): Desired acceleration

        Returns:
        - tau (Tensor): Control forces [Fx, Fy, Mz]
        """
        eta = six2threeDOF(torch.tensor(state["eta"], dtype=torch.float32))  # Current position
        nu = torch.tensor(state["nu"], dtype=torch.float32)  # Current velocity

        # Compute tracking error
        error_eta = eta_d - eta  # Position error
        error_nu = nu_d - nu  # Velocity error

        # PD Control Law (Basic Feedback)
        tau_fb = self.Kp * error_eta + self.Kd * error_nu

        # Model-Based Compensation:
        # τ = M * η_ddot + D * ν + PD-feedback
        tau_model = self.M_inv @ (eta_d_ddot - self.D @ nu)

        # Total Control Torque
        tau = tau_fb + tau_model

        return tau
