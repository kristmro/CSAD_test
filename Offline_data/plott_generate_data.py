#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the pickle file
pkl_filename = 'training_data.pkl'
with open(pkl_filename, 'rb') as f:
    raw = pickle.load(f)

# Extract training data (assumed to be a list of trajectories)
training_data = raw['training_data']
# Choose one trajectory (here, the first one)
traj = training_data[0]
# Extract the eta array (assumed shape: (T, 3))
eta = np.array(traj['eta'])
nu = np.array(traj['nu'])
tau = np.array(traj['tau_cmd'])
time = np.array(traj['time'])
# # Plot eta[1] vs eta[2]
# plt.figure(figsize=(8, 6))
# plt.plot(eta[:, 1], eta[:, 0], label='Trajectory')
# plt.xlabel('eta[1]')
# plt.ylabel('eta[0]')
# plt.title('Trajectory: eta[1] vs eta[0]')
# plt.legend()
# plt.grid(True)
# plt.show()

#printing nicly the nu and eta
print("eta: ", eta)
print("nu: ", nu)
print("tau: ", tau)
print("time: ", time)
print(f"The amount of timesteps times the dimensions of (eta+num)*time steps: {6*len(time)}")