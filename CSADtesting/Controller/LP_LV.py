from MCSimPython.utils import Rz
import numpy as np

class LP_LV:
    def __init__(self, dt, gamma_1, gamma_2, M, D, C_func):
        self.dt = dt
        self.gamma_1 = np.diag(gamma_1)  # Ensure diagonal matrix
        self.gamma_2 = np.diag(gamma_2)
        self.M = M  # Inertia matrix (Body frame)
        self.D = D  # Damping matrix (Body frame)
        self.C_func = C_func  # Coriolis matrix function (velocity-dependent)
    
    def S(self, r):
        """Skew-symmetric matrix for rotation rate r."""
        return np.array([
            [0, -r, 0],
            [r, 0, 0],
            [0, 0, 0]
        ])
    
    def get_tau(self, eta, eta_d, nu, nu_d, acc_d):
        """
        Compute control forces using backstepping.
        
        Args:
            eta (array[3]): Current pose [x, y, psi] in Earth frame.
            eta_d (array[3]): Desired pose [x_d, y_d, psi_d] in Earth frame.
            nu (array[3]): Current velocity [u, v, r] in Body frame.
            nu_d (array[3]): Desired velocity [u_d, v_d, r_d] in Earth frame.
            acc_d (array[3]): Desired acceleration [u_dot_d, v_dot_d, r_dot_d] in Earth frame.
        """
        psi = eta[-1]
        R = Rz(psi)  # Earth to Body rotation
        R_T = R.T    # Body to Earth rotation

        # Position error (rotated to Body frame)
        eta_err = np.array(eta) - np.array(eta_d)
        z_1 = R_T @ eta_err  # Error in Body frame

        # Virtual control law (Body frame)
        alpha = R_T @ nu_d - self.gamma_1 @ z_1

        # Velocity error (Body frame)
        z_2 = nu - alpha

        # Coriolis matrix (velocity-dependent, Body frame)
        C = self.C_func(nu)

        # Compute alpha_dot (derivative of virtual control)
        nu_d_body = R_T @ nu_d  # Desired velocity in Body frame
        acc_d_body = R_T @ acc_d  # Desired acceleration in Body frame
        alpha_dot = (
            acc_d_body
            + self.S(nu[-1]) @ nu_d_body  # Centripetal term
            - self.gamma_1 @ (self.S(nu[-1]) @ z_1 - self.gamma_1 @ z_1 + z_2)
        )

        # Control law (Body frame)
        tau = (
            self.M @ alpha_dot
            + C @ alpha
            + self.D @ alpha
            - self.gamma_2 @ z_2
        )

        return tau