#!/usr/bin/env python3
"""
Meta-training with Reptile in JAX for the adaptive controller and boat dynamics.
This script loads a reference trajectory (positions, velocities, and accelerations)
from a pickle file and then uses Reptile to update the meta‐parameters:
  - Controller tuning gains (K1, K2)
  - Adaptive parameters: gamma_diag (the diagonal entries of the gamma matrix) and theta_init
  - Initial estimates for the uncertain boat dynamics (here interpreted as v_init and a_init)

The dummy forward model predicts the vessel state via a second-order model:
    predicted_position(t) = initial_state + t*v_init + 0.5*t^2*a_init
    predicted_velocity(t) = v_init + t*a_init
    predicted_acceleration(t) = a_init
The loss is the mean squared error versus the desired trajectory.

Note: The adaptive gain parameter is stored as a vector `gamma_diag` (of length 3*(2N+1)).
When used in your controller you would reconstruct it as:
    gamma = jnp.diag(gamma_diag)
which guarantees that the structure remains that of a diagonal matrix.

Author: Kristian Magnus Roen 
Date: 12.02.2025
"""

import pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad
from jax.tree_util import tree_map

# --------------------------
# Utility functions for parameter updates

def load_trajectory_data(filename):
    """
    Load reference trajectory data from a pickle file.
    The pickle file is expected to contain a list of dictionaries,
    each with keys: 'time', 'eta_d', 'eta_d_dot', 'eta_d_ddot', etc.
    
    Returns:
      times: JAX array of shape [T]
      eta_d: desired positions, shape [T, 3]
      eta_d_dot: desired velocities, shape [T, 3]
      eta_d_ddot: desired accelerations, shape [T, 3]
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    # Sort data by time (if not already sorted)
    data_sorted = sorted(data, key=lambda d: d['time'])
    times = jnp.array([d['time'] for d in data_sorted])
    eta_d = jnp.array([d['eta_d'] for d in data_sorted])
    eta_d_dot = jnp.array([d['eta_d_dot'] for d in data_sorted])
    eta_d_ddot = jnp.array([d['eta_d_ddot'] for d in data_sorted])
    return times, eta_d, eta_d_dot, eta_d_ddot

def init_meta_params(N):
    """
    Initialize meta-parameters.
    We initialize:
      - K1, K2: 3-element vectors (for a 3-DOF system)
      - gamma_diag: a vector of length 3*(2N+1) whose entries will form the diagonal
                    of the adaptive gain matrix. (Default value 0.4.)
      - theta_init: zeros of size 3*(2N+1)
      - D_init: used as v_init (nominal velocity, a 3-vector)
      - disturbance_init: used as a_init (nominal acceleration, a 3-vector)
    """
    params = {
        'K1': jnp.array([0.1, 0.1, 0.1]),
        'K2': jnp.array([0.1, 0.1, 0.1]),
        # Instead of storing a full matrix, we store its diagonal as a vector.
        'gamma_diag': jnp.array([0.4] * (3 * (2 * N + 1))),
        'theta_init': jnp.zeros(3 * (2 * N + 1)),
        # Here we reinterpret D_init as the nominal velocity and 
        # disturbance_init as the nominal acceleration.
        'D_init': jnp.array([0.5, 0.5, 0.5]),
        'disturbance_init': jnp.array([0.0, 0.0, 0.0])
    }
    return params

# --------------------------
# Dummy forward (simulation) model.
# This simple model uses a second-order kinematic model:
#   predicted_position(t) = initial_state + t*v_init + 0.5*t^2*a_init
#   predicted_velocity(t) = v_init + t*a_init
#   predicted_acceleration(t) = a_init (constant)
def forward(params, times, initial_state):
    # For clarity, interpret:
    #   v_init = params['D_init']
    #   a_init = params['disturbance_init']
    v_init = params['D_init']
    a_init = params['disturbance_init']
    # Compute predicted position: shape [T, 3]
    predicted_position = initial_state + times[:, None] * v_init + 0.5 * (times**2)[:, None] * a_init
    # Compute predicted velocity: shape [T, 3]
    predicted_velocity = v_init + times[:, None] * a_init
    # Predicted acceleration is constant, replicate for each time step.
    predicted_acceleration = jnp.tile(a_init, (times.shape[0], 1))
    return predicted_position, predicted_velocity, predicted_acceleration

def loss_fn(params, times, desired_position, desired_velocity, desired_acceleration, initial_state):
    """
    Compute the loss as the mean squared error between the predicted trajectory
    (position, velocity, acceleration) and the desired values.
    """
    pred_position, pred_velocity, pred_acceleration = forward(params, times, initial_state)
    loss_pos = jnp.mean((pred_position - desired_position) ** 2)
    loss_vel = jnp.mean((pred_velocity - desired_velocity) ** 2)
    loss_acc = jnp.mean((pred_acceleration - desired_acceleration) ** 2)
    return loss_pos + loss_vel + loss_acc

# --------------------------
# Inner-loop adaptation: run a few gradient steps on a given task.
def inner_loop_update(params, times, desired_position, desired_velocity, desired_acceleration, initial_state, inner_lr, inner_steps):
    adapted_params = params
    for _ in range(inner_steps):
        grads = grad(loss_fn)(adapted_params, times, desired_position, desired_velocity, desired_acceleration, initial_state)
        adapted_params = tree_map(lambda p, g: p - inner_lr * g, adapted_params, grads)
    return adapted_params

# --------------------------
# Main meta-training loop using Reptile.
def main():
    # Load the reference trajectory from the pickle file.
    filename = 'trajectory_data_02_450_008.pkl'
    times, eta_d, eta_d_dot, eta_d_ddot = load_trajectory_data(filename)
    print(f"Loaded trajectory with {len(times)} time steps.")

    # Use the first desired position as the initial state.
    initial_state = eta_d[0]

    # Number of frequency components (as used in your controller).
    N = 15
    meta_params = init_meta_params(N)

    # Meta-training hyperparameters.
    meta_lr = 1e-3      # outer-loop (Reptile) learning rate
    inner_lr = 1e-2     # inner-loop learning rate
    inner_steps = 5     # number of inner-loop updates per task
    num_meta_iterations = 100

    # For demonstration, we simulate multiple tasks by perturbing the desired trajectory.
    # Here we assume we have multiple tasks by adding noise to position, velocity, and acceleration.
    num_tasks = 5

    for meta_iter in range(num_meta_iterations):
        # Initialize accumulator for the parameter differences.
        meta_update = {k: jnp.zeros_like(v) for k, v in meta_params.items()}
        total_loss = 0.0

        for t in range(num_tasks):
            # Create a slightly different task by adding small noise.
            key = jax.random.PRNGKey(meta_iter * 100 + t)
            noise_pos = jax.random.normal(key, eta_d.shape) * 0.01
            noise_vel = jax.random.normal(key, eta_d_dot.shape) * 0.01
            noise_acc = jax.random.normal(key, eta_d_ddot.shape) * 0.01

            desired_position = eta_d + noise_pos
            desired_velocity = eta_d_dot + noise_vel
            desired_acceleration = eta_d_ddot + noise_acc

            # Inner-loop adaptation for this task.
            adapted_params = inner_loop_update(meta_params, times, desired_position, desired_velocity, desired_acceleration, initial_state, inner_lr, inner_steps)

            # Accumulate the difference between the task-adapted and meta-parameters.
            diff = tree_map(lambda a, b: a - b, adapted_params, meta_params)
            meta_update = tree_map(lambda acc, d: acc + d, meta_update, diff)

            # Evaluate the loss for this task using the adapted parameters.
            task_loss = loss_fn(adapted_params, times, desired_position, desired_velocity, desired_acceleration, initial_state)
            total_loss += task_loss

        # Average the parameter differences and task loss over tasks.
        meta_update = tree_map(lambda x: x / num_tasks, meta_update)
        avg_loss = total_loss / num_tasks

        # Reptile meta-update: update meta parameters toward the adapted parameters.
        meta_params = tree_map(lambda p, delta: p + meta_lr * delta, meta_params, meta_update)

        print(f"Meta iteration {meta_iter+1}/{num_meta_iterations}, Avg Loss: {avg_loss:.6f}")

    # Save the final meta-parameters to a pickle file.
    # Convert JAX arrays to NumPy arrays for pickling.
    meta_params_np = tree_map(lambda x: np.array(x), meta_params)
    with open('meta_params.pkl', 'wb') as f:
        pickle.dump(meta_params_np, f)
    print("Meta parameters saved to 'meta_params.pkl'.")

if __name__ == '__main__':
    main()
