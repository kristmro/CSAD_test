#!/usr/bin/env python3
"""
trainDiff_PyTorch.py

Replicates Spencer's approach using DiffSim and PyTorch’s functional API.
For seed in [0..9] and M in [2, 5, 10, 20, 30, 40, 50]:
  - Set random seed.
  - Sub-sample M trajectories from the DiffSim-generated pickle file.
  - Shuffle time dimension per trajectory.
  - Split into training and validation sets.
  - Train an ensemble (one model per trajectory) in parallel using torch.func.vmap and functorch.grad.
  - Save the trained ensemble.
  
Author: Kristian Magnus Roen (adapted from Spencer M. Richards)
Date:   2025-02-17
"""

import os, sys, time, pickle, warnings
from typing import NamedTuple
import numpy as np
from math import pi
import tqdm.auto as tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch import functional as F

# Import functorch modules (for vmap/grad)
from torch.func import vmap, grad, functional_call

# DiffSim imports
from DiffSim.DiffUtils import six2threeDOF, Rz_torch_2
from DiffSim.Simulator.DiffCsad import DiffCSAD_6DOF as vessel

# Instantiate simulator (global constant)
csad = vessel(dt=0.08, method="RK4")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

##############################################################################
# 1) Data Processing
##############################################################################
def load_data(pkl_filename='training_data_diff_dt0.08_simtime50.0_traj500.pkl'):
    # Load the pickle file
    with open(pkl_filename, 'rb') as f:
        raw = pickle.load(f)
    traj_list = raw['training_data']
    print(f"Loaded {len(traj_list)} trajectories.")
    
    time_list, eta_list, nu_list, tau_list = [], [], [], []
    for i, d in enumerate(traj_list):
        time_tensor = torch.tensor(d['time'], dtype=torch.float32)
        eta_tensor  = torch.tensor(d['eta'], dtype=torch.float32)
        nu_tensor   = torch.tensor(d['nu'], dtype=torch.float32)
        tau_tensor  = torch.tensor(d['tau_cmd'], dtype=torch.float32)
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
    # (If nu already has 3 columns, no further indexing is needed.)
    x      = torch.cat([eta_3dof,     nu_stacked[:, :-1, :]], dim=-1)   # (num_traj, T-1, 6)
    x_next = torch.cat([eta_3dof_next, nu_stacked[:, 1:, :]],  dim=-1)     # (num_traj, T-1, 6)
    u      = tau_stacked[:, :-1]      # (num_traj, T-1, 3)
    
    print(f"x shape: {x.shape}")
    print(f"x_next shape: {x_next.shape}")
    print(f"u shape: {u.shape}")
    
    # Move all tensors to device
    data = {'t': time_stacked[:, :-1],
            't_next': time_stacked[:, 1:],
            'x': x,
            'x_next': x_next,
            'u': u}
    data = torch.utils._pytree.tree_map(lambda a: a.to(device), data)
    return data, num_dof, num_traj, num_samples

def split_shuffle_sample(data, hparams, num_traj):
    if hparams['num_subtraj'] > num_traj:
        warnings.warn(f"Cannot sub-sample {hparams['num_subtraj']} trajectories! Capping at {num_traj}.")
        hparams['num_subtraj'] = num_traj
    # Create index tensor on device
    shuffled_idx = torch.randperm(num_traj, device=device)
    hparams['subtraj_idx'] = shuffled_idx[:hparams['num_subtraj']]
    data = torch.utils._pytree.tree_map(lambda a: torch.index_select(a, 0, hparams["subtraj_idx"]), data)
    return data

##############################################################################
# 2) Batched ODE and RK38 Integrator (Ensemble Training)
##############################################################################
# (We assume the known prior dynamics come from DiffSim’s csad, which we import elsewhere.)
# Also, we assume that an appropriate batched rotation function is used.

def Rz_torch_2(psi):
    """
    Batched rotation matrix about the z-axis.
    psi: tensor of shape (...), output: shape (..., 3, 3)
    """
    c = torch.cos(psi)
    s = torch.sin(psi)
    zero = torch.zeros_like(c)
    one  = torch.ones_like(c)
    row1 = torch.stack([c, -s, zero], dim=-1)
    row2 = torch.stack([s,  c, zero], dim=-1)
    row3 = torch.stack([zero, zero, one], dim=-1)
    return torch.stack([row1, row2, row3], dim=-2)

def ode_batched(x, t, u, params):
    # Split state into eta and nu:
    eta = x[..., :3]  # (..., 3)
    nu  = x[..., 3:]  # (..., 3)
    psi = eta[..., -1]  # (...,)
    Rz = Rz_torch_2(psi)  # (..., 3, 3)
    eta_d = torch.matmul(Rz, nu.unsqueeze(-1)).squeeze(-1)  # (..., 3)
    
    # Neural network residual dynamics
    f = x.clone()
    for i, (W, b) in enumerate(zip(params['W'], params['b'])):
        f = torch.tanh(torch.matmul(f, W.transpose(-1, -2)) + b)
    f = torch.matmul(f, params['A'].transpose(-1, -2))
    
    # Prior dynamics: ensure matrices are on the same device as x.
    M_mat = six2threeDOF(csad._M).to(x.device)  # (3,3)
    D_mat = six2threeDOF(csad._D).to(x.device)  # (3,3)
    G_mat = six2threeDOF(csad._G).to(x.device)  # (3,3)
    batch_shape = x.shape[:-1]
    M_batch = M_mat.expand(*batch_shape, 3, 3)
    D_batch = D_mat.expand(*batch_shape, 3, 3)
    G_batch = G_mat.expand(*batch_shape, 3, 3)
    
    nu_unsq = nu.unsqueeze(-1)
    eta_unsq = eta.unsqueeze(-1)
    Dnu = torch.matmul(D_batch, nu_unsq).squeeze(-1)
    Geta = torch.matmul(G_batch, eta_unsq).squeeze(-1)
    
    rhs = u + f - Dnu - Geta
    nu_dot = torch.linalg.solve(M_batch, rhs.unsqueeze(-1)).squeeze(-1)
    
    return torch.cat([eta_d, nu_dot], dim=-1)


def rk38_step_batched(func, h, x, t, *args):
    """
    Batched RK38 integration step.
    h: time step, tensor of shape matching batch, or scalar.
    x: state, shape (..., state_dim)
    t: time, shape (...) matching batch dims.
    *args: additional arguments (e.g. control input u, params)
    """
    batch_shape = x.shape[:-1]
    if not torch.is_tensor(h):
        h = torch.tensor(h, dtype=x.dtype, device=x.device)
    if h.dim() == 0:
        h = h.expand(*batch_shape)
    
    s = 4
    dev = x.device
    dtype = x.dtype
    A = torch.tensor([[0, 0, 0, 0],
                      [1/3, 0, 0, 0],
                      [-1/3, 1, 0, 0],
                      [1, -1, 1, 0]], device=dev, dtype=dtype)
    b = torch.tensor([1/8, 3/8, 3/8, 1/8], device=dev, dtype=dtype)
    c = torch.tensor([0, 1/3, 2/3, 1], device=dev, dtype=dtype)
    
    K = []
    for i in range(s):
        ti = t + h * c[i]
        xi = x.clone()
        for j in range(i):
            xi = xi + h.unsqueeze(-1) * A[i, j] * K[j]
        ki = func(xi, ti, *args)
        K.append(ki)
    weighted_sum = sum(b[i] * K[i] for i in range(s))
    x_next = x + h.unsqueeze(-1) * weighted_sum
    return x_next

def tree_normsq(x_tree):
    leaves, _ = torch.utils._pytree.tree_flatten(x_tree)
    return sum(torch.sum(leaf ** 2) for leaf in leaves)

##############################################################################
# 3) Functional Adam Optimizer (Simple Implementation)
##############################################################################
from typing import NamedTuple
class AdamState(NamedTuple):
    params: any
    m: any
    v: any
    t: torch.Tensor
    lr: torch.Tensor

def adam_init(params, lr):
    m = torch.utils._pytree.tree_map(lambda p: torch.zeros_like(p), params)
    v = torch.utils._pytree.tree_map(lambda p: torch.zeros_like(p), params)
    return AdamState(params, m, v, torch.tensor(0, device=device, dtype=torch.int64),
                     torch.tensor(lr, device=device, dtype=torch.float32))

def adam_update(state: AdamState, grads, beta1=0.9, beta2=0.999, eps=1e-8):
    t = state.t + 1
    new_m = torch.utils._pytree.tree_map(lambda m, g: beta1 * m + (1-beta1)*g, state.m, grads)
    new_v = torch.utils._pytree.tree_map(lambda v, g: beta2 * v + (1-beta2)*(g**2), state.v, grads)
    m_hat = torch.utils._pytree.tree_map(lambda m: m/(1-beta1**t.item()), new_m)
    v_hat = torch.utils._pytree.tree_map(lambda v: v/(1-beta2**t.item()), new_v)
    new_params = torch.utils._pytree.tree_map(
        lambda p, m_h, v_h: p - state.lr * m_h / (torch.sqrt(v_h)+eps),
        state.params, m_hat, v_hat)
    return AdamState(new_params, new_m, new_v, t, state.lr)

def init_opt_fn(lr):
    def init_fn(params):
        return adam_init(params, lr)
    return init_fn

def update_opt_fn():
    def update_fn(idx, grads, state):
        return adam_update(state, grads)
    return update_fn

def get_params_fn():
    def get_fn(state):
        return state.params
    return get_fn

init_opt = init_opt_fn(lr=1e-3)
update_opt = update_opt_fn()
get_params = get_params_fn()

##############################################################################
# 4) Loss Function and Training Step
##############################################################################
def loss_fn(params, regularizer, t, x, u, t_next, x_next):
    # Here, t, x, u, etc. have shape [num_models, T, ...] (for each ensemble model)
    # We flatten the time dimension.
    num_models, T = x.shape[0], x.shape[1]
    x_flat      = x.reshape(num_models * T, -1)         # (B,6)
    t_flat      = t.reshape(num_models * T)             # (B,)
    u_flat      = u.reshape(num_models * T, -1)           # (B,3)
    t_next_flat = t_next.reshape(num_models * T)          # (B,)
    x_next_flat = x_next.reshape(num_models * T, -1)       # (B,6)
    dt = t_next_flat - t_flat
    # Use RK38 to compute one-step prediction for each sample.
    x_next_est_flat = rk38_step_batched(ode_batched, dt, x_flat, t_flat, u_flat, params)
    loss_val = torch.sum((x_next_flat - x_next_est_flat)**2)
    num_samples = t_flat.numel()
    return (loss_val + regularizer * tree_normsq(params)) / num_samples

def step(idx, opt_state, regularizer, batch):
    params = get_params(opt_state)
    grad_loss = grad(loss_fn, argnums=0)
    grads = grad_loss(params, regularizer, **batch)
    new_state = update_opt(idx, grads, opt_state)
    return new_state

##############################################################################
# 5) Epoch Iterator (Over the time dimension)
##############################################################################
def epoch(data, batch_size, batch_axis=1, ragged=False):
    # Assume data is a pytree with the time dimension along axis=1.
    batch_dims = torch.utils._pytree.tree_map(lambda x: x.shape[batch_axis], data)
    batch_dim_sizes = torch.utils._pytree.tree_leaves(batch_dims)
    if not all(size == batch_dim_sizes[0] for size in batch_dim_sizes):
        raise ValueError("Batch dimensions not equal!")
    num_samples = batch_dim_sizes[0]
    num_batches = (num_samples + batch_size - 1) // batch_size if ragged else num_samples // batch_size
    shuffled_indices = torch.randperm(num_samples, device=device)
    for i in range(num_batches):
        batch_idx = shuffled_indices[i * batch_size : (i + 1) * batch_size]
        batch = torch.utils._pytree.tree_map(lambda x: torch.index_select(x, batch_axis, batch_idx), data)
        yield batch

##############################################################################
# 6) Main Training Loop (Ensemble Training)
##############################################################################
def main():
    seeds = range(10)
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
            
            # Hyperparameters
            hparams = {
                'seed': seed,
                'num_subtraj': M,
                'ensemble': {
                    'num_hlayers': 2,
                    'hdim': 32,
                    'train_frac': 0.75,
                    'batch_frac': 0.25,
                    'regularizer_l2': 1e-4,
                    'learning_rate': 1e-3,
                    'num_epochs': 1000,
                }
            }
            
            # Load and process data
            data, num_dof, num_traj, num_samples = load_data(file)
            data = split_shuffle_sample(data, hparams, num_traj)
            # Data now has shapes:
            #   t, t_next: (M, T)
            #   x, x_next: (M, T, 6)
            #   u: (M, T, 3)
            num_train_samples = int(hparams['ensemble']['train_frac'] * (data['t'].shape[1]))
            ensemble_train_data = torch.utils._pytree.tree_map(lambda a: a[:, :num_train_samples], data)
            ensemble_valid_data = torch.utils._pytree.tree_map(lambda a: a[:, num_train_samples:], data)
            
            num_hlayers = hparams['ensemble']['num_hlayers']
            hdim = hparams['ensemble']['hdim']
            num_dof = 3  # as extracted from your data

            if num_hlayers >= 1:
                shapes = [(hdim, 2*num_dof)] + (num_hlayers-1)*[(hdim, hdim)]
            else:
                shapes = []

            ensemble = {
                'W': [0.1 * torch.randn(shape, device=device) for shape in shapes],
                'b': [0.1 * torch.randn((shape[0],), device=device) for shape in shapes],
                'A': 0.1 * torch.randn((3, hdim), device=device)
            }

            
            # Initialize optimizer states for the ensemble.
            # (Here we treat the network as a single model; for a true ensemble, extend dimensions accordingly.)
            from torch.utils._pytree import tree_map
            opt_state = init_opt(ensemble)
            
            # Prepare a batch iterator over the time dimension.
            batch_size = int(hparams['ensemble']['batch_frac'] * num_train_samples)
            batch = next(epoch(ensemble_train_data, batch_size, batch_axis=1, ragged=False))
            
            # Pre-compile (warm-up) the training step.
            step_idx = 0
            _ = step(step_idx, opt_state, hparams['ensemble']['regularizer_l2'], batch)
            
            # Optionally, compute a validation loss before training.
            print("ENSEMBLE TRAINING: Starting gradient descent...")
            num_epochs = hparams['ensemble']['num_epochs']
            for epoch_idx in tqdm.tqdm(range(num_epochs), desc="Epochs"):
                # Calculate number of batches per epoch for the inner progress bar
                num_batches = (num_train_samples + batch_size - 1) // batch_size
                batch_iter = epoch(ensemble_train_data, batch_size, batch_axis=1, ragged=False)
                
                # Use tqdm for the inner batch loop with dynamic description showing epoch
                for batch in tqdm.tqdm(batch_iter, desc=f"Epoch {epoch_idx+1}/{num_epochs}", 
                                      total=num_batches, leave=False):
                    opt_state = step(step_idx, opt_state, hparams['ensemble']['regularizer_l2'], batch)
                    step_idx += 1
                
                # Optional: Add validation loss calculation at the end of each epoch
                if epoch_idx % 10 == 0:
                    params = get_params(opt_state)
                    valid_loss = loss_fn(params, 0.0, **ensemble_valid_data)
                    print(f"Epoch {epoch_idx}: Validation loss = {valid_loss:.6f}")
            
            # Save results
            output_name = f"results_seed={seed}_M={M}.pkl"
            results = {
                'hparams': hparams,
                'ensemble': get_params(opt_state)
            }
            output_dir = os.path.join('train_results', 'ours')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_name)
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()
