#!/usr/bin/env python3
"""
PyTorch implementation of ensemble training on vessel dynamics using torch.func.
This implementation handles parallel training of multiple models using vmap,
with optimizations for improved performance.

Based on the work by Spencer M. Richards
"""

import os, sys, time, warnings
import numpy as np
from math import pi
import tqdm.auto as tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, grad
import torch.utils._pytree as pytree
import pickle

# DiffSim imports
from DiffSim.DiffUtils import six2threeDOF
from DiffSim.Simulator.DiffCsad import DiffCSAD_6DOF as vessel

# Check CUDA availability and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    warnings.warn("CUDA is not available! Using CPU, which will be much slower.")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# Print PyTorch version for debugging
print(f"PyTorch version: {torch.__version__}")

# Instantiate simulator
csad = vessel(dt=0.08, method="RK4")

# Check if torch.compile is available (PyTorch 2.0+)
ENABLE_COMPILE = hasattr(torch, 'compile')
if ENABLE_COMPILE:
    print("torch.compile is available - will use JIT compilation")
else:
    print("torch.compile is not available - using standard PyTorch")

# Define activation function (compatible with older PyTorch versions)
def swish(x):
    """Swish/SiLU activation function: x * sigmoid(x)"""
    return x * torch.sigmoid(x)

##############################################################################
# 1) Neural Network Definition (Functional)
##############################################################################
def nn_forward(params, x):
    """Functional version of neural network forward pass"""
    # Process all hidden layers
    h = x
    for i in range(len(params['W'])):
        h = torch.tanh(torch.nn.functional.linear(h, params['W'][i], params['b'][i]))
    
    # Last layer - final hidden layer to output, no activation
    return torch.nn.functional.linear(h, params['A'], None)

##############################################################################
# 2) Physics-Based ODE Function
##############################################################################
def ship_dynamics(x, t, u, params):
    """
    Compute ship dynamics with neural network residual
    x: state vector [eta; nu] (position/orientation and velocities)
    u: control input
    params: neural network parameters
    """
    # Extract eta and nu
    eta = x[..., :3]  # position/orientation
    nu = x[..., 3:]   # velocities
    psi = eta[..., -1]  # yaw angle
    
    # Calculate residual from neural network
    f_residual = nn_forward(params, x)
    
    # Build rotation matrix (simplified for batch operations)
    c = torch.cos(psi)
    s = torch.sin(psi)
    zero = torch.zeros_like(c)
    one = torch.ones_like(c)
    
    # Create rotation matrix elements
    row1 = torch.stack([c, -s, zero], dim=-1)
    row2 = torch.stack([s, c, zero], dim=-1)
    row3 = torch.stack([zero, zero, one], dim=-1)
    Rz = torch.stack([row1, row2, row3], dim=-2)
    
    # Kinematics
    eta_dot = torch.matmul(Rz, nu.unsqueeze(-1)).squeeze(-1)
    
    # Dynamics matrices
    M_mat = six2threeDOF(csad._M).to(device)
    D_mat = six2threeDOF(csad._D).to(device)
    G_mat = six2threeDOF(csad._G).to(device)
    
    # Expand matrices for batch operations
    batch_shape = x.shape[:-1]
    M_batch = M_mat.expand(*batch_shape, 3, 3)
    D_batch = D_mat.expand(*batch_shape, 3, 3)
    G_batch = G_mat.expand(*batch_shape, 3, 3)
    
    # Precompute inverse of M_batch for efficiency
    M_inv_batch = torch.linalg.inv(M_batch)
    
    # Compute forces
    Dnu = torch.matmul(D_batch, nu.unsqueeze(-1)).squeeze(-1)
    Geta = torch.matmul(G_batch, eta.unsqueeze(-1)).squeeze(-1)
    
    # Compute acceleration using precomputed inverse
    rhs = u + f_residual - Dnu - Geta
    nu_dot = torch.matmul(M_inv_batch, rhs.unsqueeze(-1)).squeeze(-1)
    
    # Combine kinematics and dynamics
    return torch.cat([eta_dot, nu_dot], dim=-1)

##############################################################################
# 3) RK38 Integrator Step
##############################################################################
def rk38_step(ode, dt, x, t, u, params):
    """RK38 integrator step (optimized for vmap)"""
    # RK3/8 coefficients
    A = torch.tensor([[0, 0, 0, 0],
                    [1/3, 0, 0, 0],
                    [-1/3, 1, 0, 0],
                    [1, -1, 1, 0]], device=device)
    b = torch.tensor([1/8, 3/8, 3/8, 1/8], device=device)
    c = torch.tensor([0, 1/3, 2/3, 1], device=device)
    
    # Ensure dt is properly shaped for broadcasting
    dt_expanded = dt.unsqueeze(-1) if dt.ndim == 0 else dt[..., None]
    
    # Prepare for integration
    K = []
    
    # Calculate K stages
    for i in range(4):
        # Time update
        ti = t + dt * c[i]
        
        # State update
        xi = x.clone()
        for j in range(i):
            xi = xi + dt_expanded * A[i, j] * K[j]
        
        # Calculate dynamics
        k_i = ode(xi, ti, u, params)
        K.append(k_i)
    
    # Combine stages
    weighted_sum = sum(b[i] * K[i] for i in range(4))
    x_next = x + dt_expanded * weighted_sum
    
    return x_next

##############################################################################
# 4) Loss Function
##############################################################################
def loss_fn(params, t, x, u, t_next, x_next_true, regularizer=0.0):
    """Loss function with explicit L2 regularization"""
    # Calculate dt
    dt = t_next - t
    
    # Forward pass with RK38
    x_next_pred = rk38_step(ship_dynamics, dt, x, t, u, params)
    
    # MSE prediction loss
    pred_loss = torch.mean((x_next_pred - x_next_true)**2)
    
    # Calculate L2 regularization
    reg_term = 0.0
    for param_group in params.values():
        if isinstance(param_group, list):
            for p in param_group:
                reg_term = reg_term + torch.sum(p**2)
        else:
            reg_term = reg_term + torch.sum(param_group**2)
    
    # Combined loss
    total_loss = pred_loss + regularizer * reg_term
    
    return total_loss

##############################################################################
# 5) Optimizers
##############################################################################
from typing import NamedTuple

class AdamState(NamedTuple):
    """State for functional Adam optimizer"""
    params: dict
    m: dict
    v: dict
    t: torch.Tensor
    lr: torch.Tensor

def init_adam(params, lr):
    """Initialize Adam optimizer state"""
    # Zero momentum tensors with same structure as params
    m = pytree.tree_map(lambda p: torch.zeros_like(p), params)
    v = pytree.tree_map(lambda p: torch.zeros_like(p), params)
    
    return AdamState(
        params, m, v,
        torch.tensor(0, device=device, dtype=torch.int64),
        torch.tensor(lr, device=device, dtype=torch.float32)
    )

def adam_update(state: AdamState, grads, beta1=0.9, beta2=0.999, eps=1e-8):
    """Update parameters with Adam optimizer"""
    t = state.t + 1
    
    # Update biased first moment estimate
    m = pytree.tree_map(lambda m_old, g: beta1 * m_old + (1 - beta1) * g,
                         state.m, grads)
    
    # Update biased second raw moment estimate
    v = pytree.tree_map(lambda v_old, g: beta2 * v_old + (1 - beta2) * (g**2),
                         state.v, grads)
    
    # Bias correction
    m_hat = pytree.tree_map(lambda m_t: m_t / (1 - beta1**t.item()), m)
    v_hat = pytree.tree_map(lambda v_t: v_t / (1 - beta2**t.item()), v)
    
    # Update parameters
    params = pytree.tree_map(
        lambda p, m_h, v_h: p - state.lr * m_h / (torch.sqrt(v_h) + eps),
        state.params, m_hat, v_hat
    )
    
    return AdamState(params, m, v, t, state.lr)

##############################################################################
# 6) Data Processing Functions
##############################################################################
def load_data(pkl_filename='training_data_diff_dt0.08_simtime50.0_traj500.pkl'):
    """Load and preprocess data"""
    # Load the pickle file
    with open(pkl_filename, 'rb') as f:
        raw = pickle.load(f)
    traj_list = raw['training_data']
    print(f"Loaded {len(traj_list)} trajectories.")
    
    time_list, eta_list, nu_list, tau_list = [], [], [], []
    for i, d in enumerate(traj_list):
        time_tensor = torch.tensor(d['time'], dtype=torch.float32, device=device)
        eta_tensor  = torch.tensor(d['eta'], dtype=torch.float32, device=device)
        nu_tensor   = torch.tensor(d['nu'], dtype=torch.float32, device=device)
        tau_tensor  = torch.tensor(d['tau_cmd'], dtype=torch.float32, device=device)
        time_list.append(time_tensor)
        eta_list.append(eta_tensor)
        nu_list.append(nu_tensor)
        tau_list.append(tau_tensor)
        if i < 1:
            print(f"Trajectory {i}:")
            print(f"  time shape: {time_tensor.shape}")
            print(f"  eta shape:  {eta_tensor.shape}")
            print(f"  nu shape:   {nu_tensor.shape}")
            print(f"  tau_cmd shape: {tau_tensor.shape}")
    
    time_stacked = torch.stack(time_list, dim=0)   # (num_traj, T)
    eta_stacked  = torch.stack(eta_list,  dim=0)     # (num_traj, T, 6)
    nu_stacked   = torch.stack(nu_list,   dim=0)     # (num_traj, T, 3)
    tau_stacked  = torch.stack(tau_list,  dim=0)     # (num_traj, T, 3)
    
    print(f"Stacked time shape: {time_stacked.shape}")
    print(f"Stacked eta shape:  {eta_stacked.shape}")
    print(f"Stacked nu shape:   {nu_stacked.shape}")
    print(f"Stacked tau_cmd shape: {tau_stacked.shape}")
    
    num_traj   = time_stacked.shape[0]
    T          = time_stacked.shape[1]
    num_samples = T - 1
    num_dof    = nu_stacked.shape[-1]
    
    # Use columns [0,1,5] from eta for the 3DOF state
    eta_3dof      = eta_stacked[:, :-1, [0,1,5]]   # (num_traj, T-1, 3)
    eta_3dof_next = eta_stacked[:, 1:,  [0,1,5]]     # (num_traj, T-1, 3)
    
    # Concatenate eta (3) and nu (3) to form state x of dimension 6
    x      = torch.cat([eta_3dof,     nu_stacked[:, :-1, :]], dim=-1)   # (num_traj, T-1, 6)
    x_next = torch.cat([eta_3dof_next, nu_stacked[:, 1:, :]],  dim=-1)     # (num_traj, T-1, 6)
    u      = tau_stacked[:, :-1]      # (num_traj, T-1, 3)
    
    print(f"x shape: {x.shape}")
    print(f"x_next shape: {x_next.shape}")
    print(f"u shape: {u.shape}")
    
    # Create data dictionary with current and next states
    data = {'t': time_stacked[:, :-1],
            't_next': time_stacked[:, 1:],
            'x': x,
            'x_next': x_next,
            'u': u}
    
    return data, num_dof, num_traj, num_samples

def split_shuffle_sample(data, hparams, num_traj):
    """Select a subset of trajectories based on hparams"""
    M = hparams['num_subtraj']
    if M > num_traj:
        warnings.warn(f"Cannot sub-sample {M} trajectories! Capping at {num_traj}.")
        M = num_traj
        hparams['num_subtraj'] = M
    
    # Create index tensor on device
    key = torch.randperm(num_traj, device=device)
    selected_idx = key[:M]
    hparams['subtraj_idx'] = selected_idx.cpu().numpy().tolist()
    
    # Select the trajectories - use tree_map to ensure consistency
    sampled_data = pytree.tree_map(
        lambda a: torch.index_select(a, 0, selected_idx),
        data
    )
    
    return sampled_data

def split_train_valid(data, train_frac):
    """Split data into training and validation sets (along time dimension)"""
    num_timesteps = data['t'].shape[1]
    num_train_samples = int(train_frac * num_timesteps)
    
    # Split each trajectory into train and validation sets
    train_data = pytree.tree_map(
        lambda a: a[:, :num_train_samples],
        data
    )
    
    valid_data = pytree.tree_map(
        lambda a: a[:, num_train_samples:],
        data
    )
    
    return train_data, valid_data

def get_batch(data, batch_size):
    """Get a random batch from the data (preserving temporal relationships)"""
    num_traj = data['t'].shape[0]
    num_timesteps = data['t'].shape[1]
    
    # Generate random start indices for each trajectory
    # Ensure we don't go out of bounds
    max_start = num_timesteps - batch_size
    if max_start <= 0:
        # If we can't fit a batch, use the whole trajectory
        batch = data
        return batch
    
    starts = torch.randint(0, max_start + 1, (num_traj,), device=device)
    
    # Select contiguous chunks for each trajectory
    batch = {}
    for k, v in data.items():
        # Initialize tensor for batches
        batch_v = torch.zeros((num_traj, batch_size) + v.shape[2:], 
                              dtype=v.dtype, device=device)
        
        # Select chunk for each trajectory
        for i in range(num_traj):
            start = starts[i]
            end = start + batch_size
            batch_v[i] = v[i, start:end]
            
        batch[k] = batch_v
    
    return batch

##############################################################################
# 7) Ensemble Training
##############################################################################
def init_models(num_models, num_hlayers, hdim):
    """Initialize ensemble model parameters with proper shapes"""
    input_dim = 6  # [eta, nu] - 3DOF position/orientation and 3DOF velocities
    output_dim = 3  # residual term (3 DOF)
    
    # Initialize params dictionary
    ensemble = {
        'W': [],  # Hidden layer weights
        'b': [],  # Hidden layer biases
    }
    
    # Input to first hidden layer
    shapes = []
    shapes.append((hdim, input_dim))  # First layer (input -> hidden)
    
    # Hidden to hidden layers
    for _ in range(num_hlayers - 1):
        shapes.append((hdim, hdim))  # Hidden -> hidden layers
    
    # Initialize hidden layers
    for i in range(num_hlayers):
        ensemble['W'].append(0.1 * torch.randn(num_models, shapes[i][0], shapes[i][1], device=device))
        ensemble['b'].append(0.1 * torch.randn(num_models, shapes[i][0], device=device))
    
    # Output layer (no bias)
    ensemble['A'] = 0.1 * torch.randn(num_models, output_dim, hdim, device=device)
    
    return ensemble

def train_ensemble(data, M, hparams):
    """Train ensemble of M models using torch.func for parallel computation"""
    # Split data into train and validation sets WITHOUT shuffling time
    train_data, valid_data = split_train_valid(data, hparams['train_frac'])
    
    # Initialize ensemble model parameters with proper shapes
    ensemble_params = init_models(
        num_models=M,
        num_hlayers=hparams['num_hlayers'],
        hdim=hparams['hdim']
    )
    
    # Initialize optimizer state
    opt_state = init_adam(ensemble_params, hparams['learning_rate'])
    
    # Track best models
    best_params = pytree.tree_map(lambda p: p.clone(), ensemble_params)
    best_losses = torch.full((M,), float('inf'), device=device)
    best_steps = torch.zeros(M, dtype=torch.long, device=device)
    step_idx = 0
    
    # Vectorize loss and gradient calculations across models
    vmapped_loss = vmap(loss_fn, in_dims=(0, None, 0, 0, None, 0, None))
    vmapped_grad = vmap(grad(loss_fn), in_dims=(0, None, 0, 0, None, 0, None))
    
    # Print initial validation loss
    print("Computing initial validation loss...", end="", flush=True)
    with torch.no_grad():
        valid_losses = vmapped_loss(
            opt_state.params,
            valid_data['t'],
            valid_data['x'],
            valid_data['u'],
            valid_data['t_next'],
            valid_data['x_next'],
            0.0  # No regularization for evaluation
        )
        mean_valid_loss = torch.mean(valid_losses)
    
    print(f"\nInitial validation loss: {mean_valid_loss.item():.6f}")
    print(f"Individual losses: min={torch.min(valid_losses).item():.6f}, "
          f"max={torch.max(valid_losses).item():.6f}, "
          f"std={torch.std(valid_losses).item():.6f}")
    
    # Training loop
    print("Starting training...")
    batch_size = int(hparams['batch_frac'] * train_data['t'].shape[1])
    num_epochs = hparams['num_epochs']
    regularizer = hparams['regularizer_l2']
    
    # Single progress bar for epochs
    epoch_pbar = tqdm.tqdm(range(num_epochs), desc="Training")
    
    for epoch in epoch_pbar:
        epoch_losses = []
        
        # Number of batches per epoch
        num_batches = max(1, train_data['t'].shape[1] // batch_size)
        
        # Train on mini-batches - no nested progress bar
        for _ in range(num_batches):
            # Get a batch
            batch = get_batch(train_data, batch_size)
            
            # Compute gradients in parallel across models and samples
            grads = vmapped_grad(
                opt_state.params,
                batch['t'],
                batch['x'],
                batch['u'],
                batch['t_next'],
                batch['x_next'],
                regularizer
            )
            
            # Update parameters with Adam optimizer
            opt_state = adam_update(opt_state, grads)
            step_idx += 1
            
            # Compute batch loss (without regularization for reporting)
            with torch.no_grad():
                losses = vmapped_loss(
                    opt_state.params,
                    batch['t'],
                    batch['x'],
                    batch['u'],
                    batch['t_next'],
                    batch['x_next'],
                    0.0  # No regularization for evaluation
                )
                epoch_losses.append(torch.mean(losses).item())
        
        # Check validation loss and update best models
        if epoch % 5 == 0 or epoch == num_epochs - 1:  # Check more frequently
            with torch.no_grad():
                # Compute validation loss for each model
                valid_losses = vmapped_loss(
                    opt_state.params,
                    valid_data['t'],
                    valid_data['x'],
                    valid_data['u'],
                    valid_data['t_next'],
                    valid_data['x_next'],
                    0.0  # No regularization for evaluation
                )
                
                # Check if any models improved and update them
                improved = False
                for i in range(M):
                    if valid_losses[i] < best_losses[i]:
                        improved = True
                        # Update parameters for this specific model index
                        for key, value in opt_state.params.items():
                            if isinstance(value, list):
                                for j in range(len(value)):
                                    best_params[key][j][i] = opt_state.params[key][j][i].clone()
                            else:
                                best_params[key][i] = opt_state.params[key][i].clone()
                        
                        best_losses[i] = valid_losses[i].item()
                        best_steps[i] = step_idx
                
                # Update progress bar description
                mean_valid_loss = torch.mean(valid_losses).item()
                mean_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
                
                if improved:
                    status = "✓"
                else:
                    status = "✗"
                    
                epoch_pbar.set_description(
                    f"Epoch {epoch}: Train={mean_train_loss:.6f}, Valid={mean_valid_loss:.6f} {status}"
                )
                
                # Only print full stats occasionally to reduce output clutter
                if epoch % 20 == 0 or epoch == num_epochs - 1:
                    print(f"\nEpoch {epoch}: Train loss = {mean_train_loss:.6f}, Validation loss = {mean_valid_loss:.6f}")
                    print(f"Individual losses: min={torch.min(valid_losses).item():.6f}, "
                          f"max={torch.max(valid_losses).item():.6f}, "
                          f"std={torch.std(valid_losses).item():.6f}")
                    print(f"Best steps: {best_steps}")
                    if improved:
                        print(f"✓ Found better models!")
                    else:
                        print(f"✗ No improvement")
    
    return best_params, best_losses.cpu().numpy(), best_steps.cpu().numpy()

##############################################################################
# 8) Main Function
##############################################################################
def main():
    # Make code deterministic if needed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    seeds = range(10)  # 0-9
    M_values = [2, 5, 10, 20, 30, 40, 50]
    file = 'training_data_diff_dt0.08_simtime50.0_traj500.pkl'
    
    for seed in seeds:
        for M in M_values:
            print(f"\n\n===========================")
            print(f"   SEED = {seed},   M = {M} ")
            print("===========================")
            
            # Set random seeds
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            
            # Define hyperparameters
            hparams = {
                'seed': seed,
                'num_subtraj': M,
                'num_hlayers': 2,
                'hdim': 32,
                'train_frac': 0.75,
                'batch_frac': 0.25,
                'regularizer_l2': 1e-4,
                'learning_rate': 1e-2,  # Match Spencer's learning rate
                'num_epochs': 1000,
            }
            
            # Load and prepare data
            data, num_dof, num_traj, num_samples = load_data(file)
            sampled_data = split_shuffle_sample(data, hparams, num_traj)
            
            # Train models in parallel
            best_params, best_losses, best_steps = train_ensemble(sampled_data, M, hparams)
            
            # Save results
            output_name = f"seed={seed}_M={M}.pkl"
            output_dir = os.path.join('train_results', 'ours')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_name)
            
            # Prepare results dictionary
            results = {
                'hparams': hparams,
                'best_step_idx': best_steps.tolist(),
                'ensemble': best_params,
                'best_losses': best_losses.tolist()
            }
            
            # Use standard pickle for older PyTorch versions
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()