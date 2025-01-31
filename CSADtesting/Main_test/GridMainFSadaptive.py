import sys
import numpy as np
import time
from MCSimPython.utils import six2threeDOF

sys.path.append('./CSADtesting')
from Environment.GridBoatEnv import GridWaveEnvironment
from Controller.adaptiveFScontroller import AdaptiveFSController  

def compute_desired_trajectory(t, goal_n, goal_e, current_eta, dt):
    """Compute desired pose and derivatives (static goal example)."""
    current_n, current_e, current_yaw = current_eta
    desired_yaw = np.arctan2(goal_e - current_e, goal_n - current_n)
    eta_d = np.array([goal_n, goal_e, desired_yaw])
    eta_d_dot = np.zeros(3)
    eta_d_ddot = np.zeros(3)
    return eta_d, eta_d_dot, eta_d_ddot

def main():
    # Environment setup
    env = GridWaveEnvironment(
        dt=0.1, grid_width=15, grid_height=6, render_on=True, final_plot=True
    )
    env.set_task(
        start_position=(2, 2, 90),
        goal=(4, 12, 1),
        wave_conditions=(1, 4.5, 0),
        goal_func=None,
        obstacle_func=None,
        obstacles=None
    )


    M_6DOF = env.vessel._M
    D_6DOF = env.vessel._D


    # Initialize controller with converted matrices
    controller = AdaptiveFSController(dt=0.1, M=M_6DOF, D=D_6DOF, N=15)
    controller.set_tuning_params(
        K1=[0.005, 0.005, 0.001],
        K2=[0.30, 0.30, 0.5],
        gamma=[1e-3]*( (2*15 +1)*3 )
    )

    # Simulation loop
    simtime = 150.0
    max_steps = int(simtime / 0.1)
    for step in range(max_steps):
        state = env.get_state()
        boat_n, boat_e = state["boat_position"]
        boat_yaw_rad = state["boat_orientation"]
        nu = state["velocities"]
        goal_n, goal_e, _ = state["goal"]

        # Current state in radians
        eta = np.array([boat_n, boat_e, boat_yaw_rad])

        # Desired trajectory
        eta_d, eta_d_dot, eta_d_ddot = compute_desired_trajectory(
            env.simulation_time, goal_n, goal_e, eta, env.dt
        )

        # Compute control action
        tau = controller.get_tau(
            eta=eta,
            eta_d=eta_d,
            nu=nu,
            eta_d_dot=eta_d_dot,
            eta_d_ddot=eta_d_ddot,
            t=env.simulation_time
        )
        print(tau)

        # Apply action (surge, sway, yaw)
        _, done, _, _ = env.step(tau)
        
        if done:
            break

    env.plot_trajectory()

if __name__ == "__main__":
    main()