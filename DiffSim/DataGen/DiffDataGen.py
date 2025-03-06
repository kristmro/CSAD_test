#!/usr/bin/env python3
"""
Script to generate training data by simulating a boat in waves with varying wave conditions
using the DiffSim pipeline and a model-based PD controller.

Scaling applied:
  - λ = 1/90
  - Time scaled by sqrt(λ)
  - Wave height scaled by λ

Wave conditions (full-scale before scaling):
  - Wave direction: uniformly sampled from 0° to 90°
  - Wave height: uniformly sampled between 0 and 5
  - Wave period: sampled from a truncated normal distribution in [6, 20] with mean increasing with wave height

Author: Kristian Magnus Roen
Date: 19.02.2025
"""

import torch
import numpy as np
import pickle
from scipy.stats import truncnorm

# Import DiffSim modules
from DiffSim.Environment.DiffGridBoatEnv import DiffGridBoatEnvironment
from DiffSim.Controller.DiffModelController import DiffModelController
from DiffSim.Allocation.DiffAllocation import PseudoInverseAllocator
from DiffSim.Allocation.DiffThrusterDynamics import ThrusterDynamics, ThrusterData
from DiffSim.Allocation.DiffThruster import Thruster

# -----------------------------
# Constants and Scaling Factors
# -----------------------------
LAMBDA = 1 / 90          # Model scaling factor
TIME_SCALE = np.sqrt(LAMBDA)  # Time scaling factor
WAVE_HEIGHT_SCALE = LAMBDA      # Wave height scaling

def truncated_normal_samples(mean_array, lower, upper, std=1.0):
    """
    Vectorized function to sample from a truncated normal distribution for each element in mean_array.
    
    Parameters:
      mean_array : 1D numpy array of means.
      lower      : Lower bound for truncation.
      upper      : Upper bound for truncation.
      std        : Standard deviation of the underlying normal.
    
    Returns:
      1D numpy array of samples.
    """
    alpha = (lower - mean_array) / std
    beta  = (upper - mean_array) / std
    samples = truncnorm.rvs(a=alpha, b=beta, loc=mean_array, scale=std, size=len(mean_array))
    return samples

def run_simulation(wave_conditions, dt, simtime, start_pos):
    """
    Runs a single simulation trajectory using the DiffSim pipeline with the specified wave conditions.
    
    Parameters:
      wave_conditions : tuple (scaled_wave_height, scaled_wave_period, wave_direction)
      dt              : Time step (seconds)
      simtime         : Total simulation time (seconds)
      start_pos       : Tuple (north, east, heading) for the boat's start position
      
    Returns:
      sim_data : dict containing simulation history and the wave conditions.
    """
    # Initialize environment (rendering disabled for data generation)
    env = DiffGridBoatEnvironment(
        dt=dt,
        grid_width=15,
        grid_height=6,
        render_on=False,
        final_plot=False
    )
    env.set_task(
        start_pos,
        goal=None,
        wave_conditions=wave_conditions,
        four_corner_test=True,
        simtime=simtime,
        ref_omega=[0.2, 0.2, 0.1]
    )
    
    # Initialize the model-based PD controller
    controller = DiffModelController(dt=dt)
    
    # Initialize thrust allocation
    allocator = PseudoInverseAllocator()
    for i in range(6):
        allocator.add_thruster(Thruster(pos=[ThrusterData.lx[i], ThrusterData.ly[i]], K=ThrusterData.K[i]))
    
    thrust_dynamics = ThrusterDynamics()
    u_stored = [torch.zeros(6)]
    
    # Storage lists for simulation data
    time_history    = []
    eta_history     = []
    nu_history      = []
    tau_history     = []
    tau_cmd_history = []
    
    steps = int(simtime / dt)
    for step_count in range(steps):
        current_time = step_count * dt
        time_history.append(current_time)
        
        # Get the desired reference trajectory from the four-corner test
        eta_d, nu_d, eta_d_ddot, _ = env.get_four_corner_nd(step_count)
        state = env.get_state()
        
        # Compute control forces with the model-based PD controller
        tau = controller.compute_control(state, eta_d, nu_d, eta_d_ddot)
        tau_history.append(tau.detach().cpu().numpy())
        
        # Allocate thrust using the pseudo-inverse method
        u, alpha = allocator.allocate(tau)
        u_stored.append(u)
        u = thrust_dynamics.limit_rate(u, u_stored[-2], ThrusterData.alpha_dot_max, dt)
        u = thrust_dynamics.saturate(u, ThrusterData.thruster_min, ThrusterData.thrust_max)
        tau_cmd = thrust_dynamics.get_tau(u, alpha)
        tau_cmd_history.append(tau_cmd.detach().cpu().numpy())
        
        # Record state; assume state has keys 'eta' and 'nu'
        eta_history.append(state["eta"])
        nu_history.append(state["nu"])
        
        # Step the simulation
        _, done, info, _ = env.step(action=tau_cmd)
        if done:
            print("Simulation ended early:", info)
            break
    
    # Package simulation data
    sim_data = {
        'time':         np.array(time_history),
        'eta':          np.array(eta_history),
        'nu':           np.array(nu_history),
        'tau':          np.array(tau_history),
        'tau_cmd':      np.array(tau_cmd_history),
        'wave_conditions': wave_conditions,
    }
    return sim_data

def main():
    dt = 0.08           # Time step in seconds
    simtime = 50.0     # Total simulation time in seconds
    start_pos = (2.0, 2.0, 0.0)  # Starting position (north, east, heading)
    
    num_trajectories = 500  # Number of simulation runs (adjust as needed)
    training_data = []
    
    # Set random seed for reproducibility
    np.random.seed(0)
    
    for traj in range(num_trajectories):
        # --- Sample full-scale wave conditions for this trajectory ---
        # Wave direction: uniformly between 0° and 90°
        wave_direction = round(np.random.uniform(0, 90), 0)
        
        # Wave height (full scale): uniformly between 0 and 5
        wave_height_full = np.random.uniform(0, 5)
        # Scale wave height for the model
        scaled_wave_height = round(wave_height_full * WAVE_HEIGHT_SCALE,3)
        
        # Mean wave period (full scale) linearly interpolated between 6 s and 20 s
        period_mean_full = 6 + (wave_height_full / 5.0) * (20 - 6)
        # Sample wave period from a truncated normal distribution in [6, 20]
        wave_period_full = truncated_normal_samples(
            mean_array=np.array([period_mean_full]),
            lower=6.0,
            upper=20.0,
            std=1.0
        )[0]
        # Scale wave period for the model
        scaled_wave_period = round(wave_period_full * TIME_SCALE,3)
        
        # Pack the scaled wave conditions: (scaled_wave_height, scaled_wave_period, wave_direction)
        wave_conditions = (scaled_wave_height, scaled_wave_period, wave_direction)
        print(f"Trajectory {traj+1}/{num_trajectories}: wave_conditions = {wave_conditions}")
        
        # Run the simulation for this trajectory
        sim_data = run_simulation(wave_conditions, dt, simtime, start_pos)
        training_data.append(sim_data)
    
    # Save the collected training data to a pickle file
    output = {
        'training_data': training_data,
    }
    file_name = f"training_data_diff_dt{dt}_simtime{simtime}_traj{num_trajectories}.pkl"
    with open(file_name, 'wb') as f:
        pickle.dump(output, f)
    print(f"Training data saved to '{file_name}'.")

if __name__ == "__main__":
    main()
