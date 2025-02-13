#!/usr/bin/env python3
"""
Script to generate training data by simulating a boat in waves with varying wave conditions.
Each trajectory uses a random set of wave conditions:
  - Wave direction sampled uniformly from 0° to 90°,
  - Wave height sampled uniformly between 0 and 5, and
  - Wave period chosen from a normal distribution whose mean increases with wave height.
  
In addition to the usual state and control data, this script records the adaptive
controller’s tuning parameters and the evolution of its adaptive parameter estimates.

Author: Kristian Magnus Roen
Date: 12.02.2025
"""

import sys
import numpy as np
import time
import pickle

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
    max_steps = int(simtime / dt)
    
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
    M = CSAD_DP_6DOF(dt)._M
    D = CSAD_DP_6DOF(dt)._D
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
    theta_history   = []  # adaptive parameter estimates

    # Thruster allocation and dynamics
    allocator = al.PseudoInverseAllocator()
    for i in range(6):
        allocator.add_thruster(thruster.Thruster(pos=[data.lx[i], data.ly[i]], K=data.K[i]))
    ThrustDyn = dynamics.ThrusterDynamics()
    u_stored = [np.zeros(6)]
    
    # Main simulation loop
    for step_count in range(int(simtime/dt)):
        current_time = step_count * dt
        time_history.append(current_time)
        
        # Get desired reference states from the environment
        eta_d, nu_d, eta_d_ddot, nu_d_body = env.get_four_corner_nd(step_count)
        state = env.get_state()
        nu_d = Rz(state["eta"][-1]) @ nu_d
        
        # Use calculate_bias=True so that the controller returns (tau, debug)
        tau, debug = controller.get_tau(
            eta=six2threeDOF(state["eta"]),
            eta_d=eta_d,
            nu=state["nu"],
            eta_d_dot=nu_d,
            eta_d_ddot=eta_d_ddot,
            t=current_time,
            calculate_bias=True  # returns both tau and debug info
        )
        tau_history.append(tau)
        
        # Record adaptive parameter history
        theta_history.append(controller.get_theta())
        
        # Allocate control forces to thrusters and apply dynamics
        u, alpha = allocator.allocate(tau)
        u_stored.append(u)
        u = ThrustDyn.limit_rate(u, u_stored[-2], data.alpha_dot_max, dt)
        u = ThrustDyn.saturate(u, data.thruster_min, data.thrust_max)
        tau_cmd = ThrustDyn.get_tau(u, alpha)
        tau_cmd_history.append(tau_cmd)
        
        eta_history.append(state["eta"])
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
        'theta_history':   np.array(theta_history),
        'controller_tuning': {
            'K1': tuning_K1,
            'K2': tuning_K2,
            'gamma': tuning_gamma,
            'theta_bound': controller._theta_bound,
            'N': controller._N,
            'freqs': controller._freqs,
        },
        'wave_conditions': wave_conditions,
    }
    return sim_data


def main():
    # Simulation parameters
    dt      = 0.08      # Time step (seconds)
    simtime = 450      # Total simulation time (seconds)
    start_pos = (2, 2, 0)  # Initial boat position (x, y, heading)
    
    num_trajectories = 50  # Number of simulation runs (trajectories) to generate
    training_data = []
    
    # Set random seed for reproducibility
    np.random.seed(0)
    
    for traj in range(num_trajectories):
        # --- Sample wave conditions for this trajectory ---
        # Wave direction: uniformly between 0° and 90°
        wave_direction = np.random.uniform(0, 90)
        # Wave height (Hs): uniformly between 0 and 5
        wave_height = np.random.uniform(0, 5)
        # Compute a mean wave period based on the wave height.
        # For example, linearly interpolate such that:
        #   - Hs = 0   corresponds to period = 6 seconds,
        #   - Hs = 5   corresponds to period = 20 seconds.
        period_mean = 6 + (wave_height / 5.0) * (20 - 6)
        # Sample the wave period from a normal distribution around the computed mean
        wave_period = np.random.normal(period_mean, 1.0)  # Standard deviation of 1 sec
        # Clip the period so it always lies between 6 and 20 seconds
        wave_period = np.clip(wave_period, 6, 20)
        
        # Pack the wave conditions into a tuple: (Hs, Tp, wave_direction)
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
    with open('training_data.pkl: '+str(time), 'wb') as f:
        pickle.dump(output, f)
    print("Training data saved to 'training_data.pkl'.")


if __name__ == "__main__":
    main()

