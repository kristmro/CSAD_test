#!/usr/bin/env python3
"""
Script to generate training data by simulating a boat in waves with varying wave conditions.
Each trajectory uses a random set of wave conditions:
  - Wave direction sampled uniformly from 0° to 90°,
  - Wave height sampled uniformly between 0 and 5 (full scale), and
  - Wave period sampled from a truncated normal distribution whose mean
    increases with wave height (between 6 s and 20 s).

Author: Kristian Magnus Roen
Date: 12.02.2025
"""

import sys
import numpy as np
import pickle
from scipy.stats import truncnorm

# Adjust path if needed so Python finds your modules
sys.path.append('./CSADtesting')

# Import simulation environment, controller, and other necessary modules
from CSADtesting.Environment.GridBoatEnv import GridWaveEnvironment
from CSADtesting.Controller.adaptiveFScontroller import AdaptiveFSController
from MCSimPython.utils import Rz, six2threeDOF, three2sixDOF
from MCSimPython.simulator.csad import CSAD_DP_6DOF
import CSADtesting.allocation.allocation as al
import MCSimPython.simulator.thruster_dynamics as dynamics
import MCSimPython.thrust_allocation.thruster as thruster
import MCSimPython.vessel_data.CSAD.thruster_data as data


def truncated_normal_samples(mean_array, lower, upper, std=1.0):
    """
    Vectorized function to sample from a truncated normal distribution
    for each element in mean_array.

    :param mean_array: 1D numpy array of means (one for each sample).
    :param lower:      Lower bound of truncation.
    :param upper:      Upper bound of truncation.
    :param std:        Standard deviation for the underlying normal.
    :return:           1D numpy array of samples with the same shape as mean_array.
    """
    alpha = (lower - mean_array) / std
    beta  = (upper - mean_array) / std
    samples = truncnorm.rvs(a=alpha, b=beta, loc=mean_array, scale=std, size=len(mean_array))
    return samples


def run_simulation(wave_conditions, dt, simtime, start_pos):
    """
    Run a single simulation trajectory with the specified wave_conditions.

    Parameters:
        wave_conditions (tuple): (wave_height, wave_period, wave_direction)
        dt (float): Simulation time step (seconds)
        simtime (float): Total simulation time (seconds)
        start_pos (tuple): Starting position (x, y, heading) for the boat

    Returns:
        sim_data (dict): Dictionary containing the simulation history, wave conditions,
                         and adaptive controller tuning and parameter history.
    """
    # Create the simulation environment (disable rendering for training data generation)
    env = GridWaveEnvironment(
        dt=dt,
        grid_width=15,
        grid_height=6,
        render_on=False,    # Disable on-screen rendering for batch runs
        final_plot=False    # No final plot
    )
    env.set_task(
        start_position=start_pos,
        wave_conditions=wave_conditions,
        four_corner_test=True,
        simtime=simtime
    )

    # Instantiate the controller
    csad_dp = CSAD_DP_6DOF(dt)
    M = csad_dp._M
    D = csad_dp._D
    N = 512  # or your desired number of frequency components
    controller = AdaptiveFSController(dt=dt, M=M, D=D, N=N)

    # Set explicit tuning parameters if desired
    tuning_K1 = [0.1, 0.1, 0.1]
    tuning_K2 = [0.1, 0.1, 0.1]
    tuning_gamma = [0.4] * ((2 * controller._N + 1) * 3)
    controller.set_tuning_params(tuning_K1, tuning_K2, tuning_gamma)

    # Initialize storage lists
    time_history    = []
    eta_history     = []
    nu_history      = []
    tau_history     = []
    tau_cmd_history = []
    #theta_history   = []  # adaptive parameter estimates

    # Thruster allocation and dynamics
    allocator = al.PseudoInverseAllocator()
    for i in range(6):
        allocator.add_thruster(thruster.Thruster(pos=[data.lx[i], data.ly[i]], K=data.K[i]))
    ThrustDyn = dynamics.ThrusterDynamics()
    u_stored = [np.zeros(6)]

    # Main simulation loop
    steps = int(simtime / dt)
    for step_count in range(steps):
        current_time = step_count * dt
        time_history.append(current_time)

        # Get desired reference states from the environment
        eta_d, nu_d, eta_d_ddot, _ = env.get_four_corner_nd(step_count)
        state = env.get_state()
        nu_d = Rz(state["eta"][-1]) @ nu_d

        # Use calculate_bias=True if you want the controller to return (tau, debug)
        tau = controller.get_tau(
            eta=six2threeDOF(state["eta"]),
            eta_d=eta_d,
            nu=state["nu"],
            eta_d_dot=nu_d,
            eta_d_ddot=eta_d_ddot,
            t=current_time,
            calculate_bias=False
        )
        tau_history.append(tau)

        #theta_history.append(controller.get_theta())  # If you want to store adaptive parameters

        # Allocate control forces to thrusters and apply dynamics
        u, alpha = allocator.allocate(tau)
        u_stored.append(u)
        u = ThrustDyn.limit_rate(u, u_stored[-2], data.alpha_dot_max, dt)
        u = ThrustDyn.saturate(u, data.thruster_min, data.thrust_max)
        tau_cmd = ThrustDyn.get_tau(u, alpha)
        tau_cmd_history.append(tau_cmd)

        eta_history.append(six2threeDOF(state["eta"]))
        nu_history.append(state["nu"])

        # Advance simulation step
        _, done, info, _ = env.step(action=tau_cmd)
        if done:
            print("Simulation ended early:", info)
            break

    # Package simulation data (convert lists to arrays as needed)
    sim_data = {
        'time':            np.array(time_history),
        'eta':             np.array(eta_history),
        'nu':              np.array(nu_history),
        'tau':             np.array(tau_history),
        'tau_cmd':         np.array(tau_cmd_history),
        # 'theta_history':   np.array(theta_history),
        'wave_conditions': wave_conditions,
    }
    return sim_data


def main():
    dt       = 0.08  # Time step (seconds)
    simtime  = 40.0  # Total simulation time (seconds)
    start_pos = (2, 2, 0)  # Initial boat position (x, y, heading)
    
    num_trajectories = 500  # Number of simulation runs (trajectories) to generate
    training_data = []

    # Set random seed for reproducibility
    np.random.seed(0)

    for traj in range(num_trajectories):
        # --- Sample wave conditions for this trajectory ---
        # 1) Wave direction: uniform in [0, 90]
        wave_direction = round(np.random.uniform(0, 90), 2)

        # 2) Wave height (Hs): uniform in [0, 5]
        wave_height = round(np.random.uniform(0, 5), 2)

        # 3) Mean wave period: linearly interpolated from 6 s (Hs=0) to 20 s (Hs=5)
        period_mean = 6 + (wave_height / 5.0) * (20 - 6)

        # 4) Sample from truncated normal in [6, 20], std dev = 1.0
        wave_period_sample = truncated_normal_samples(
            mean_array=np.array([period_mean]),
            lower=6.0,
            upper=20.0,
            std=1.0
        )[0]  # because we passed an array of length 1

        wave_period = round(wave_period_sample, 2)

        # If needed, convert to model scale:
        wave_height = wave_height / 90.0
        wave_period = wave_period * np.sqrt(1/90.0)

        # Pack the wave conditions: (Hs, Tp, wave_direction)
        wave_conditions = (wave_height, wave_period, wave_direction)
        print(f"Trajectory {traj+1}/{num_trajectories}: wave_conditions = {wave_conditions}")

        # --- Run the simulation with these wave conditions ---
        sim_data = run_simulation(wave_conditions, dt, simtime, start_pos)
        training_data.append(sim_data)

    # Save the collected training data to a pickle file
    output = {
        'dt': dt,
        'simtime': simtime,
        'start_pos': start_pos,
        'num_trajectories': num_trajectories,
        'training_data': training_data,
    }
    with open('training_data_500_40.pkl', 'wb') as f:
        pickle.dump(output, f)
    print("Training data saved to 'training_data_500_40.pkl'.")


if __name__ == "__main__":
    main()
