# ------------------------------------------------------------------------
# Step 1 Import Necessary Libraries
# ------------------------------------------------------------------------

import pickle
import numpy as np
from tqdm import tqdm
from scipy.signal import lti, lsim
from MCSimPython.simulator.csad import CSAD_DP_6DOF
from MCSimPython.waves.wave_loads import WaveLoad
from MCSimPython.waves.wave_spectra import JONSWAP
from MCSimPython.utils import three2sixDOF, six2threeDOF
#from CSADtesting.filters.reference_filter import reference_forecorner, smooth_filter
#from CSADtesting.allocation.allocation import CSADThrusterAllocator

# ------------------------------------------------------------------------
# Step 2: Generate Smooth trajectories with smooth filter
# ------------------------------------------------------------------------
# Re-import necessary libraries after execution state reset
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Define Waypoints for Four-Corner Test
# ------------------------------
waypoints = np.array([
    [2, 2, 0], [4, 2, 0], [4, 4, 0],
    [4, 4, np.pi/4], [2, 4, np.pi/4], [2, 2, 0]
])

# ------------------------------
# Simulation Parameters
# ------------------------------
dt = 0.1  # Time step
total_time = 350  # seconds #some are showing they use 400 and other use 300 sec (ref ole nikolai lyngstaads: ship motion control concepts considering acutator constraints)
time_steps = int(total_time / dt)

# ------------------------------
# Reference Model Parameters (as per the provided formulation)
# ------------------------------
zeta = np.array([0.9, 0.9, 0.9])  # Damping ratios
omega = np.array([0.8, 0.8, 0.8])  # Resonance frequencies
t_f = np.array([2.4, 2.4,2.4])  # Time constants for filter

# Define system matrices as per the reference model equations
Omega = np.diag(2 * zeta * omega)  # Damping matrix
Gamma = np.diag(omega ** 2)  # Stiffness matrix
Af = np.diag(1 / t_f)  # Filter matrix

# ------------------------------
# Initialize Variables
# ------------------------------
x_ref = np.zeros((time_steps, 3))  # Filtered set-point
x_d = np.zeros((time_steps, 3))  # Desired position
v_d = np.zeros((time_steps, 3))  # Desired velocity
a_d = np.zeros((time_steps, 3))  # Desired acceleration

# Set initial values
x_ref[0] = waypoints[0]
x_d[0] = waypoints[0]

# ------------------------------
# Apply Reference Model Over Time
# ------------------------------
for i in range(1, time_steps):
    # Find the nearest waypoint based on time proportion
    waypoint_index = min(i * len(waypoints) // time_steps, len(waypoints) - 1)
    eta_r = waypoints[waypoint_index]  # New set-point

    # Compute filtered reference point
    x_ref[i] = x_ref[i - 1] + dt * (-Af @ x_ref[i - 1] + Af @ eta_r)

    # Compute acceleration from the reference model
    a_d[i] = -Omega @ v_d[i - 1] - Gamma @ x_d[i - 1] + Gamma @ x_ref[i]

    # Update velocity
    v_d[i] = v_d[i - 1] + a_d[i] * dt

    # Update position
    x_d[i] = x_d[i - 1] + v_d[i] * dt

# ------------------------------
# Time Vector and Waypoint Times
# ------------------------------
time = np.linspace(0, dt * (time_steps - 1), time_steps)
waypoint_times = np.linspace(0, dt * (time_steps - 1), len(waypoints))

# ------------------------------
# Plot: North, East, and Heading vs Time
# ------------------------------
fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# North Position
axs[0].plot(time, x_d[:, 0], label="North Position", color='b')
axs[0].scatter(waypoint_times, waypoints[:, 0], color='red', label="Waypoints", zorder=3)
axs[0].set_ylabel("North [m]")
axs[0].legend()
axs[0].grid()

# East Position
axs[1].plot(time, x_d[:, 1], label="East Position", color='g')
axs[1].scatter(waypoint_times, waypoints[:, 1], color='red', label="Waypoints", zorder=3)
axs[1].set_ylabel("East [m]")
axs[1].legend()
axs[1].grid()

# Heading
axs[2].plot(time, x_d[:, 2], label="Heading", color='orange')
axs[2].scatter(waypoint_times, waypoints[:, 2], color='red', label="Waypoints", zorder=3)
axs[2].set_ylabel("Heading [rad]")
axs[2].set_xlabel("Time [s]")
axs[2].legend()
axs[2].grid()

plt.suptitle("Reference Model with Filtered Set-Point")
plt.show()

# ------------------------------
# Plot: North vs East Trajectory
# ------------------------------
plt.figure(figsize=(8, 6))
plt.plot(x_d[:, 1], x_d[:, 0], label="Reference Model Trajectory", linewidth=2, color='purple')
plt.scatter(waypoints[:, 1], waypoints[:, 0], color='red', label="Waypoints", zorder=3)
plt.xlabel("East [m]")
plt.ylabel("North [m]")
plt.title("Reference Model Trajectory in North-East Plane")
plt.legend()
plt.grid()
plt.show()
# 2D Plot: North vs East Trajectory with Corrected Heading Indication
plt.figure(figsize=(8, 6))

# Plot trajectory
plt.plot(x_d[:, 1], x_d[:, 0], label="Reference Model Trajectory", linewidth=2, color='purple')

# Plot waypoints
plt.scatter(waypoints[:, 1], waypoints[:, 0], color='red', label="Waypoints", zorder=3)

# Add heading vectors (fixing rotation direction)
quiver_scale = 0.5  # Scale factor for arrow length
arrow_skip = 100  # Skip points to reduce clutter

for i in range(0, time_steps, arrow_skip):
    dx = np.cos(-x_d[i, 2] + np.pi / 2) * quiver_scale  # Correct for clockwise rotation
    dy = np.sin(-x_d[i, 2] + np.pi / 2) * quiver_scale
    plt.arrow(x_d[i, 1], x_d[i, 0], dx, dy, head_width=0.1, head_length=0.1, fc='blue', ec='blue')

plt.xlabel("East [m]")
plt.ylabel("North [m]")
plt.title("Reference Model Trajectory with Corrected Heading (Right-Handed)")
plt.legend()
plt.grid()
plt.show()
