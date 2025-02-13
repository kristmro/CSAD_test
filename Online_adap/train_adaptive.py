# #!/usr/bin/env python3
# """
# Meta-training with Reptile in JAX for the adaptive controller and boat dynamics.
# This script loads a reference trajectory (positions, velocities, and accelerations)
# from a pickle file and then uses Reptile to update the meta‐parameters:
#   - Controller tuning gains (K1, K2)
#   - Adaptive parameters: gamma_diag (the diagonal entries of the gamma matrix) and theta_init
#   - Initial estimates for the uncertain boat dynamics (here interpreted as v_init and a_init)

# The dummy forward model predicts the vessel state via a second-order model:
#     predicted_position(t) = initial_state + t*v_init + 0.5*t^2*a_init
#     predicted_velocity(t) = v_init + t*a_init
#     predicted_acceleration(t) = a_init
# The loss is the mean squared error versus the desired trajectory.

# Note: The adaptive gain parameter is stored as a vector `gamma_diag` (of length 3*(2N+1)).
# When used in your controller you would reconstruct it as:
#     gamma = jnp.diag(gamma_diag)
# which guarantees that the structure remains that of a diagonal matrix.

# Author: Kristian Magnus Roen 
# Date: 12.02.2025
# """

# import pickle
# import numpy as np
# import jax
# import jax.numpy as jnp
# from jax import grad
# from jax.tree_util import tree_map


# # --------------------------
# import pickle
# import numpy as np
# import jax.numpy as jnp
# from MCSimPython.utils import six2threeDOF  # your conversion function

# # Load the raw training data.
# with open('training_data.pkl', 'rb') as file:
#     raw = pickle.load(file)

# # The raw dictionary has keys:
# #   'dt', 'simtime', 'start_pos', 'num_trajectories', 'training_data'
# # where training_data is a list of trajectory dictionaries.
# training_data = raw['training_data']

# # Extract the lists for each field.
# # Each trajectory dictionary contains:
# #   'time':    np.array(time_history)      shape: (T,)
# #   'eta':     np.array(eta_history)       shape: (T, 6)   <-- 6 DOF position
# #   'nu':      np.array(nu_history)        shape: (T, 3)   <-- 3 DOF velocity
# #   'tau_cmd': np.array(tau_cmd_history)   shape: (T, control_dim)
# times_list   = [traj['time'] for traj in training_data]   # List of arrays, each shape (T,)
# eta_list     = [traj['eta'] for traj in training_data]    # Each shape (T, 6)
# nu_list      = [traj['nu'] for traj in training_data]     # Each shape (T, 3)
# tau_cmd_list = [traj['tau_cmd'] for traj in training_data] # Each shape (T, control_dim)

# # Stack into numpy arrays.
# # times: shape (num_traj, T)
# # eta: shape (num_traj, T, 6)
# # nu: shape (num_traj, T, 3)
# # tau_cmd: shape (num_traj, T, control_dim)
# times   = np.array(times_list)
# eta     = np.array(eta_list)
# nu      = np.array(nu_list)
# tau_cmd = np.array(tau_cmd_list)

# # Convert each trajectory's 6-DOF eta to 3-DOF.
# # Use np.apply_along_axis so that six2threeDOF is applied to each time step (each row).
# eta_3d = np.array([np.apply_along_axis(six2threeDOF, 1, traj_eta) for traj_eta in eta])
# # Now, eta_3d should have shape (num_traj, T, 3).

# # Now form transitions.
# # Let state x = [eta_3d, nu] (i.e. 3 + 3 = 6 dimensions).
# # For each trajectory, we form transitions using consecutive time steps.
# # Current state (x): from time indices 0 to T-2
# # Next state (x_next): from time indices 1 to T-1
# num_traj = times.shape[0]          # e.g. 50 trajectories
# T = times.shape[1]                 # e.g. 5625 time steps per trajectory
# num_samples = T - 1                # transitions per trajectory

# # Form the state transitions:
# x = np.concatenate((eta_3d[:, :-1, :], nu[:, :-1, :]), axis=-1)    # shape: (num_traj, T-1, 6)
# x_next = np.concatenate((eta_3d[:, 1:, :], nu[:, 1:, :]), axis=-1)   # shape: (num_traj, T-1, 6)

# # Current time and next time:
# t = times[:, :-1]    # shape: (num_traj, T-1)
# t_next = times[:, 1:]  # shape: (num_traj, T-1)

# # Control input at current time step.
# u = tau_cmd[:, :-1, :]   # shape: (num_traj, T-1, control_dim)

# # Optionally, convert to JAX arrays.
# t = jnp.array(t)
# x = jnp.array(x)
# u = jnp.array(u)
# t_next = jnp.array(t_next)
# x_next = jnp.array(x_next)

# # Number of degrees of freedom for eta (after conversion) is 3.
# num_dof = eta_3d.shape[-1]

# print("Number of trajectories:", num_traj)
# print("Time steps per trajectory:", T)
# print("Number of transitions per trajectory:", num_samples)
# print("Degrees of freedom (position after conversion):", num_dof)
# print("Data keys and shapes:")
# print("t:", t.shape)
# print("x:", x.shape)
# print("u:", u.shape)
# print("t_next:", t_next.shape)
# print("x_next:", x_next.shape)

# # Package the data in a dictionary if needed.
# data = {'t': t, 'x': x, 'u': u, 't_next': t_next, 'x_next': x_next}



#!/usr/bin/env python3
"""
Steps:
1. Load raw data from training_data.pkl
2. Arrange Data: extract time, 6-DOF eta (convert to 3-DOF), nu, tau_cmd; form transitions.
3. Shuffle and Sub-sample Trajectories & Split Each Trajectory into Training/Validation Sets.
4. Model Ensemble Training (using known M and D to compute nominal dynamics, then learning residual Δ).

In our setup, the physics-informed model computes:
    a_nom = M⁻¹(u - D @ nu)
and the network learns Δ so that:
    nu_next = nu + dt * (a_nom + Δ)
with the kinematics:
    eta_next = eta + dt * nu
"""

import os, sys, time, pickle
import numpy as np
from tqdm.auto import tqdm

# Adjust path so Python finds CSADtesting modules
sys.path.append('./CSADtesting')
from MCSimPython.utils import six2threeDOF  # conversion function from 6-DOF to 3-DOF

# -----------------------------------
# Step 1. Load raw data from training_data.pkl
# -----------------------------------
with open('training_data.pkl', 'rb') as f:
    raw = pickle.load(f)

# Assume the raw pickle contains a dictionary with key 'training_data'
# which is a list of trajectory dictionaries.
training_data = raw['training_data']

# -----------------------------------
# Step 2. Arrange Data
# -----------------------------------
# Extract time, eta, nu, and control commands (tau_cmd) for each trajectory.
times_list   = [traj['time'] for traj in training_data]      # Each: (T,)
eta_list     = [traj['eta'] for traj in training_data]       # Each: (T, 6)
nu_list      = [traj['nu'] for traj in training_data]        # Each: (T, 3)
tau_cmd_list = [traj['tau_cmd'] for traj in training_data]   # Each: (T, control_dim)

# Convert lists to arrays (shape: (num_traj, T, ...))
times   = np.array(times_list)    # (num_traj, T)
eta6d   = np.array(eta_list)        # (num_traj, T, 6)
nu      = np.array(nu_list)         # (num_traj, T, 3)
tau_cmd = np.array(tau_cmd_list)    # (num_traj, T, control_dim)

# Convert 6-DOF eta to 3-DOF (using six2threeDOF)
eta3d = np.array([np.apply_along_axis(six2threeDOF, 1, traj_eta)
                  for traj_eta in eta6d])   # (num_traj, T, 3)

# Form state transitions per trajectory.
# Let x = [eta3d, nu] (concatenated along last axis: 3 + 3 = 6)
# For each trajectory, define:
#    x: from indices 0 to T-2,  x_next: from indices 1 to T-1.
num_traj = times.shape[0]
T = times.shape[1]
x      = np.concatenate((eta3d[:, :-1, :], nu[:, :-1, :]), axis=-1)    # (num_traj, T-1, 6)
x_next = np.concatenate((eta3d[:, 1:, :],  nu[:, 1:, :]), axis=-1)       # (num_traj, T-1, 6)
t      = times[:, :-1]    # (num_traj, T-1)

# -----------------------------------
# Step 3. Shuffle and Sub-sample Trajectories & Split Each Trajectory
# -----------------------------------
# First, randomly shuffle the trajectories and select a subset.
np.random.seed(0)  # for reproducibility
traj_indices = np.arange(num_traj)
np.random.shuffle(traj_indices)
num_subtraj = min(50, num_traj)  # e.g., use at most 50 trajectories
selected_idx = traj_indices[:num_subtraj]

# Subsample the trajectories.
x       = x[selected_idx]       # (num_subtraj, T-1, 6)
x_next  = x_next[selected_idx]  # (num_subtraj, T-1, 6)
t       = t[selected_idx]       # (num_subtraj, T-1, ...)
u       = tau_cmd[:, :-1, :]    # (num_traj, T-1, control_dim)
u       = u[selected_idx]       # (num_subtraj, T-1, control_dim)

# Now, instead of flattening directly, we form a data dictionary.
data = {'x': x, 'u': u, 'x_next': x_next}

# Shuffle time samples _within each trajectory_ and then split into training/validation.
import jax
import jax.numpy as jnp
import jax.random as random
import jax.example_libraries.optimizers as optimizers

# Set up a JAX random key and generate one key per trajectory.
key = random.PRNGKey(42)
# Generate a key for each trajectory (num_subtraj keys)
keys = random.split(key, x.shape[0])

# Use jax.tree_util.tree_map to shuffle each trajectory along the time dimension (axis=1).
shuffled_data = jax.tree_util.tree_map(
    lambda a: jax.vmap(lambda k, arr: jax.random.permutation(k, arr))(keys, a),
    data
)

# Define the fraction of each trajectory to use for training.
train_frac = 0.75
num_samples = x.shape[1]  # T-1
num_train_samples = int(train_frac * num_samples)

ensemble_train_data = jax.tree_util.tree_map(
    lambda a: a[:, :num_train_samples, ...],
    shuffled_data
)
ensemble_valid_data = jax.tree_util.tree_map(
    lambda a: a[:, num_train_samples:, ...],
    shuffled_data
)

# For model training, flatten the time dimension.
def flatten_data(a):
    return a.reshape((-1, a.shape[-1]))

train_x      = flatten_data(ensemble_train_data['x'])
train_u      = flatten_data(ensemble_train_data['u'])
train_x_next = flatten_data(ensemble_train_data['x_next'])

val_x      = flatten_data(ensemble_valid_data['x'])
val_u      = flatten_data(ensemble_valid_data['u'])
val_x_next = flatten_data(ensemble_valid_data['x_next'])

# Determine dt from the first trajectory's time stamps.
dt = float(t[0, 1] - t[0, 0])

# -----------------------------------
# Step 4. Model Ensemble Training
# -----------------------------------
# We now train an ensemble of neural networks that learn the residual dynamics
# on top of the known physics.
# The known dynamics are given by:
#    a_nom = M⁻¹ (u - D @ nu)
# where M and D come from your simulator.
from MCSimPython.simulator.csad import CSAD_DP_6DOF
from MCSimPython.utils import six2threeDOF
sim_instance = CSAD_DP_6DOF(dt)
# Here we assume that six2threeDOF can be applied elementwise if needed.
# In many cases, M and D are 6x6 matrices; if you use 3-DOF after conversion,
# you might need to extract the relevant 3x3 block.
# For this example, we assume M_known and D_known are 3x3.
M_known = jnp.array(six2threeDOF(sim_instance._M))
D_known = jnp.array(six2threeDOF(sim_instance._D))

# Problem dimensions:
state_dim   = 6         # [eta (3) ; nu (3)]
control_dim = train_u.shape[-1]  # e.g., control_dim (e.g., 3)
input_dim   = state_dim + control_dim  # NN input: [x; u]
output_dim  = 3         # NN predicts residual acceleration correction Δ (for nu)

# -----------------------------------------------------------------------------
# Hyperparameters for Ensemble Training
# -----------------------------------------------------------------------------
hp = {
    'num_models':     10,
    'num_hlayers':    2,
    'hdim':           32,
    'regularizer_l2': 1e-4,
    'learning_rate':  1e-2,
    'num_epochs':     1000,
    'batch_frac':     0.25,
}

# -----------------------------------------------------------------------------
# Define the neural network model (for residual Δ)
# -----------------------------------------------------------------------------
def nn_forward(params, inp):
    f = inp
    for W, b in zip(params['W'], params['b']):
        f = jnp.tanh(jnp.dot(W, f) + b)
    return jnp.dot(params['A'], f)

def model_predict(params, x, u, dt, M, D):
    """
    x: current state (batch, 6) with x = [eta; nu]
    u: control input (batch, control_dim)
    Computes:
      a_nom = M⁻¹ (u - D @ nu)
      Δ = nn_forward(params, [x; u])
      a_total = a_nom + Δ
      nu_next = nu + dt * a_total
      eta_next = eta + dt * nu
    Returns concatenated next state [eta_next; nu_next]
    """
    eta = x[:, :3]
    nu  = x[:, 3:]
    a_nom = jax.vmap(lambda nu_i, u_i: jnp.linalg.solve(M, u_i - D @ nu_i))(nu, u)
    inp = jnp.concatenate([x, u], axis=-1)
    delta = jax.vmap(lambda z: nn_forward(params, z))(inp)
    a_total = a_nom + delta
    nu_next = nu + dt * a_total
    eta_next = eta + dt * nu
    return jnp.concatenate([eta_next, nu_next], axis=-1)

def loss_single(params, batch, dt, reg_l2):
    x      = batch['x']
    u      = batch['u']
    x_next = batch['x_next']
    x_next_pred = model_predict(params, x, u, dt, M_known, D_known)
    mse = jnp.mean((x_next_pred - x_next)**2)
    reg = sum([jnp.sum(W**2) for W in params['W']]) \
          + sum([jnp.sum(b**2) for b in params['b']]) \
          + jnp.sum(params['A']**2)
    return mse + reg_l2 * reg

loss_ensemble = jax.vmap(loss_single, in_axes=(0, None, None, None))

# -----------------------------------------------------------------------------
# Initialize Ensemble
# -----------------------------------------------------------------------------
num_hlayers = hp['num_hlayers']
hdim = hp['hdim']
if num_hlayers >= 1:
    layer_shapes = [(hdim, input_dim)] + (num_hlayers - 1) * [(hdim, hdim)]
else:
    layer_shapes = []

# Total keys needed = 1 + 2*num_hlayers + 1
num_keys = 1 + 2 * num_hlayers + 1

def init_params(keys_W, keys_b, key_A):
    params = {}
    params['W'] = []
    params['b'] = []
    for i in range(num_hlayers):
        W = 0.1 * random.normal(keys_W[i], (hdim, layer_shapes[i][1]))
        b = 0.1 * random.normal(keys_b[i], (hdim,))
        params['W'].append(W)
        params['b'].append(b)
    params['A'] = 0.1 * random.normal(key_A, (output_dim, hdim))
    return params

def init_ensemble(key, num_models):
    def init_single(key):
        subkeys = random.split(key, num_keys)
        keys_W = subkeys[1:1+num_hlayers]
        keys_b = subkeys[1+num_hlayers:1+2*num_hlayers]
        key_A = subkeys[-1]
        return init_params(keys_W, keys_b, key_A)
    keys_model = random.split(key, num_models)
    return jax.vmap(init_single)(keys_model)

key = random.PRNGKey(0)
ensemble = init_ensemble(key, hp['num_models'])

# -----------------------------------------------------------------------------
# Optimizer Setup
# -----------------------------------------------------------------------------
opt_init, opt_update, get_params = optimizers.adam(hp['learning_rate'])
opt_states = jax.vmap(opt_init)(ensemble)
num_train_samples = train_x.shape[0]
batch_size = int(hp['batch_frac'] * num_train_samples)

@jax.jit
def train_step(opt_state, batch, dt, reg_l2):
    def loss_fn(params):
        return loss_single(params, batch, dt, reg_l2)
    grads = jax.grad(loss_fn)(get_params(opt_state))
    return opt_update(0, grads, opt_state)

ensemble_train_step = jax.jit(jax.vmap(train_step, in_axes=(0, None, None, None)))

@jax.jit
def compute_loss(opt_state, batch, dt, reg_l2):
    return loss_single(get_params(opt_state), batch, dt, reg_l2)

ensemble_compute_loss = jax.jit(jax.vmap(compute_loss, in_axes=(0, None, None, None)))

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
print("Initial validation losses:",
      ensemble_compute_loss(opt_states,
                            {'x': jnp.array(val_x), 'u': jnp.array(val_u), 'x_next': jnp.array(val_x_next)},
                            dt, hp['regularizer_l2']))

best_opt_states = opt_states
best_losses = ensemble_compute_loss(opt_states,
                                    {'x': jnp.array(val_x), 'u': jnp.array(val_u), 'x_next': jnp.array(val_x_next)},
                                    dt, hp['regularizer_l2'])

for epoch in tqdm(range(hp['num_epochs'])):
    idx = np.random.choice(num_train_samples, batch_size, replace=False)
    mini_batch = {'x': train_x[idx],
                  'u': train_u[idx],
                  'x_next': train_x_next[idx]}
    opt_states = ensemble_train_step(opt_states, mini_batch, dt, hp['regularizer_l2'])
    val_losses = ensemble_compute_loss(opt_states,
                                       {'x': jnp.array(val_x), 'u': jnp.array(val_u), 'x_next': jnp.array(val_x_next)},
                                       dt, hp['regularizer_l2'])
    cond = val_losses < best_losses  # shape: (num_models,)
    best_opt_states = jax.tree_util.tree_map(
        lambda b, c: jnp.where(cond.reshape((-1,) + (1,)*(b.ndim - 1)), c, b),
        best_opt_states, opt_states)
    best_losses = jnp.where(cond, val_losses, best_losses)


print("Final best validation losses:", best_losses)
final_ensemble = jax.vmap(get_params)(best_opt_states)

with open('final_ensemble.pkl', 'wb') as f:
    pickle.dump(final_ensemble, f)

print("Model ensemble training complete.")
