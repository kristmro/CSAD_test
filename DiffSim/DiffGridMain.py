# example_pid_run.py
import numpy as np
import time
from DiffGridBoatEnv import DiffGridBoatEnvironment
from MCSimPython.utils import six2threeDOF

class PIDController:
    """Simple PID controller for 3DOF ship control."""
    def __init__(self, Kp, Ki, Kd, dt):
        """
        Initialize the PID controller.
        
        Parameters:
        - Kp: Proportional gain (array of 3 values for surge, sway, yaw)
        - Ki: Integral gain
        - Kd: Derivative gain
        - dt: Time step
        """
        self.Kp = np.array(Kp)
        self.Ki = np.array(Ki)
        self.Kd = np.array(Kd)
        self.dt = dt

        self.integral_error = np.zeros(3)
        self.prev_error = np.zeros(3)

    def compute_control(self, eta, eta_d, nu, nu_d):
        """
        Compute the PID control force (tau) based on errors.

        Parameters:
        - eta: Current position (north, east, yaw)
        - eta_d: Desired position (north, east, yaw)
        - nu: Current velocity (u, v, r)
        - nu_d: Desired velocity (u, v, r)

        Returns:
        - tau: Control forces (surge, sway, yaw)
        """
        error = eta_d - eta  # Position error
        d_error = (error - self.prev_error) / self.dt  # Derivative error
        self.integral_error += error * self.dt  # Integral error

        # PID output
        tau = self.Kp * error + self.Ki * self.integral_error + self.Kd * d_error

        # Store previous error for next time step
        self.prev_error = error

        return tau

def main():
    # Environment setup
    dt = 0.08  # Simulation time step
    simtime = 450  # Simulation duration in seconds
    max_steps = int(simtime / dt)

    env = DiffGridBoatEnvironment(dt=dt, grid_width=15, grid_height=6, render_on=True, final_plot=True)

    # Start position and wave conditions
    start_pos = (2.0, 2.0, 0.0)  # (north, east, heading)
    wave_cond = (0.03, 5.0, 90.0)  # (Hs=0.5m, Tp=5s, waveDir=90° => from east)

    env.set_task(start_position=start_pos, wave_conditions=wave_cond, four_corner_test=True, simtime=simtime)

    # Initialize simple PID controller
    pid = PIDController(Kp=[10.0, 10.0, 5.0],  # High proportional gain
                        Ki=[0.1, 0.1, 0.05],  # Low integral gain to prevent wind-up
                        Kd=[5.0, 5.0, 2.0],  # Derivative gain for smoothness
                        dt=dt)

    # Run the simulation
    print("Starting four-corner test with PID controller...")
    start_time = time.time()

    for step_count in range(max_steps):
        eta_d, nu_d, eta_d_ddot, nu_d_body = env.get_four_corner_nd(step_count)
        state = env.get_state()

        eta = six2threeDOF(state["eta"])  # Get (north, east, yaw)
        nu = state["nu"]  # Get (u, v, r)

        # Compute control force (tau) using PID
        tau = pid.compute_control(eta, eta_d, nu, nu_d)

        # Apply action and check if done
        _, done, info, _ = env.step(action=tau)

        if done:
            print("Simulation ended:", info)
            break

    total_time = time.time() - start_time
    print(f"Simulation completed in {total_time:.2f} seconds")
    env.plot_trajectory()

if __name__ == "__main__":
    main()
