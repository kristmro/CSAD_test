import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from CSAD.fast_adap_embedding.Envionment.wave_environment import WaveEnvironment
from CSAD.fast_adap_embedding.Controller.mbrl_controller import MBRLController
from MCSimPython.guidance.filter import ThrdOrderRefFilter
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

def generate_reference_path(sim_time, dt):
    """
    Generate a smooth reference path using ThrdOrderRefFilter.
    sim_time: Total simulation time (seconds).
    dt: Simulation timestep (seconds).
    Returns: Reference model and trajectory array.
    """
    t = np.arange(0, sim_time, dt)
    ref_model = ThrdOrderRefFilter(dt)

    # Define setpoints
    set_points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 50.0, 0.0]),
        np.array([50.0, 50.0, 0.0]),
        np.array([50.0, 50.0, -np.pi / 4]),
        np.array([50.0, 0.0, -np.pi / 4]),
        np.array([0.0, 0.0, 0.0])
    ]

    # Generate trajectory
    trajectory = np.zeros((len(t), 9))
    for i in range(1, len(t)):
        if t[i] > 500:
            ref_model.set_eta_r(set_points[5])
        elif t[i] > 400:
            ref_model.set_eta_r(set_points[4])
        elif t[i] > 300:
            ref_model.set_eta_r(set_points[3])
        elif t[i] > 200:
            ref_model.set_eta_r(set_points[2])
        elif t[i] > 100:
            ref_model.set_eta_r(set_points[1])
        else:
            ref_model.set_eta_r(set_points[0])
        ref_model.update()
        trajectory[i] = ref_model._x

    return trajectory

# Initialize environment and controller
environment = WaveEnvironment(dt=0.01)
state_dim = 3
action_dim = 3
mbrl_controller = MBRLController(state_dim, action_dim, device="cuda")

# Generate smooth reference path
sim_time = 600  # seconds
dt = 0.01  # timestep
reference_path = generate_reference_path(sim_time, dt)

# Learning Phase: Collect Data by Following the Smooth Reference Path
print("Collecting training data by following the smooth reference path...")
all_training_data = []
wave_conditions = [(4.5, 14, 180), (2.0, 8.0, 90), (6.0, 20.0, 135)]

for hs, tp, wave_dir in tqdm(wave_conditions, desc="Wave Conditions"):
    environment.set_wave_conditions(hs, tp, wave_dir)
    training_data = mbrl_controller.collect_data(environment, reference_path=reference_path[:, :3], steps_per_episode=500)
    all_training_data.append(training_data)

# Combine data for training
combined_states = torch.cat([data[0] for data in all_training_data])
combined_actions = torch.cat([data[1] for data in all_training_data])
combined_next_states = torch.cat([data[2] for data in all_training_data])

# Create PyTorch Dataset and DataLoader
dataset = TensorDataset(combined_states, combined_actions, combined_next_states)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the loss function
loss_fn = nn.MSELoss()

# Train the dynamics model
print("Training dynamics model...")
mbrl_controller.train_dynamics_model(optimizer=mbrl_controller.optimizer,
                                      loss_fn=loss_fn,
                                      data_loader=data_loader,
                                      epochs=50)

# Execution Phase with Online Model Updates
print("Running simulation to follow the reference path with MPC and online adaptation...")
execution_data = []
update_frequency = 50
step_index = 0
eta_storage = []

for step in tqdm(range(1000), desc="Simulation Steps"):
    current_state = torch.tensor(environment.get_state()[:state_dim], dtype=torch.float32).unsqueeze(0).to("cuda")

    # Stop if at the last point of the reference path
    if step_index < len(reference_path):
        action, step_index = mbrl_controller.plan_action_mpc(current_state, reference_path[:, :3], step_index, horizon=4, lambda_reg=1)
        next_state = environment.step(action.cpu().numpy())
        
        # Append current state to eta_storage
        eta_storage.append(environment.get_state()[:state_dim].tolist())

        # Collect data for online adaptation
        execution_data.append((current_state.squeeze(0).cpu().numpy(), action.cpu().numpy().squeeze(0), next_state[:state_dim]))

        # Periodically retrain the dynamics model
        if step > 0 and step % update_frequency == 0:
            print("Retraining dynamics model with online data...")
            exec_states, exec_actions, exec_next_states = zip(*execution_data)
            exec_states = torch.tensor(np.array(exec_states), dtype=torch.float32).to("cuda")
            exec_actions = torch.tensor(np.array(exec_actions), dtype=torch.float32).to("cuda")
            exec_next_states = torch.tensor(np.array(exec_next_states), dtype=torch.float32).to("cuda")

            # Combine with original training data
            all_states = torch.cat([combined_states, exec_states])
            all_actions = torch.cat([combined_actions, exec_actions])
            all_next_states = torch.cat([combined_next_states, exec_next_states])

            # Create a new DataLoader
            dataset = TensorDataset(all_states, all_actions, all_next_states)
            data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

            # Retrain the dynamics model
            mbrl_controller.train_dynamics_model(optimizer=mbrl_controller.optimizer,
                                                  loss_fn=loss_fn,
                                                  data_loader=data_loader,
                                                  epochs=5)  # Fewer epochs for online updates
            execution_data = []  # Clear execution data after retraining
    else:
        print("Reached the last point in the trajectory. Stopping...")
        break

# Convert results to array for visualization
eta_storage = np.array(eta_storage)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(eta_storage[:, 1], eta_storage[:, 0], label="Controlled Trajectory (East vs North)")
plt.plot(reference_path[:, 1], reference_path[:, 0], '--', label="Reference Path")
plt.xlabel("East (m)")
plt.ylabel("North (m)")
plt.legend()
plt.grid()
plt.title("Controlled Vessel Trajectory Following a Smooth Path with MPC")
plt.show()
