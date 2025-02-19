#!/usr/bin/env python3
"""
train_ours.py

Replicates Spencer's approach in a single Python file:
  - for seed in [0..9]
    for M in [2, 5, 10, 20, 30, 40, 50]
      * set random seed
      * sub-sample M trajectories
      * shuffle time dimension per trajectory
      * split train/valid
      * train ensemble (best model per trajectory)
      * save ensemble to results_seed={seed}_M={M}.pkl

Author: Kristian Magnus Roen
Date:   2025-02-17
"""

import os
import sys
import pickle
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from math import pi
from MCSimPython.simulator.csad import CSAD_DP_6DOF
from MCSimPython.utils import six2threeDOF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

##############################################################################
# 1) Process Data (similar to your original "process_data")
##############################################################################
def load_data(pkl_filename='training_data.pkl'):
    """
    Loads the pickle file containing a list of trajectories, each with keys:
      'time':    (T,)
      'eta':     (T, 3)
      'nu':      (T, 3)
      'tau_cmd': (T, control_dim)
    Returns a dict of Tensors shaped (num_traj, T-1, ...).
    """
    with open(pkl_filename, 'rb') as f:
        raw = pickle.load(f)
    training_data = raw['training_data']

    t_list, x_list, u_list = [], [], []
    t_next_list, x_next_list = [], []

    for traj in training_data:
        t_traj   = traj['time']    # shape (T,)
        eta_traj = traj['eta']     # shape (T, 3)
        nu_traj  = traj['nu']      # shape (T, 3)
        u_traj   = traj['tau_cmd'] # shape (T, control_dim)

        # Build "transitions" from step k to k+1
        t_list.append(t_traj[:-1])
        t_next_list.append(t_traj[1:])
        x_list.append(np.concatenate([eta_traj[:-1], nu_traj[:-1]], axis=-1))  
        x_next_list.append(np.concatenate([eta_traj[1:],  nu_traj[1:]], axis=-1))
        u_list.append(u_traj[:-1])  # shape (T-1, control_dim)

    # Convert to torch Tensors
    t      = torch.tensor(np.stack(t_list, axis=0),      dtype=torch.float32, device=device)
    t_next = torch.tensor(np.stack(t_next_list, axis=0), dtype=torch.float32, device=device)
    x      = torch.tensor(np.stack(x_list, axis=0),      dtype=torch.float32, device=device)
    x_next = torch.tensor(np.stack(x_next_list, axis=0), dtype=torch.float32, device=device)
    u      = torch.tensor(np.stack(u_list, axis=0),      dtype=torch.float32, device=device)

    return {'t': t, 't_next': t_next, 'x': x, 'x_next': x_next, 'u': u}


##############################################################################
# 2) Sub-sample M Trajectories
##############################################################################
def subsample_trajectories(data, M, seed=0):
    torch.manual_seed(seed)
    num_traj = data['t'].shape[0]
    if M > num_traj:
        warnings.warn(f"Requested M={M} but only {num_traj} available.")
        M = num_traj
    idx = torch.randperm(num_traj)[:M]
    return {k: v[idx] for k, v in data.items()}


##############################################################################
# 3) Shuffle Time Indices (per trajectory)
##############################################################################
def shuffle_time_for_each_trajectory(data, seed=0):
    """
    For each trajectory i, we create a random permutation of [0..(T-1)]
    and reorder data[i, ...] accordingly in the time dimension.
    """
    torch.manual_seed(seed)
    num_traj, T = data['t'].shape[0], data['t'].shape[1]

    shuffled = {}
    for key, val in data.items():
        # shape check
        if val.dim() == 2:  # (num_traj, T)
            out = torch.empty_like(val)
            for i in range(num_traj):
                perm_i = torch.randperm(T)
                out[i] = val[i, perm_i]
            shuffled[key] = out

        elif val.dim() == 3:  # (num_traj, T, feature_dim)
            out = torch.empty_like(val)
            for i in range(num_traj):
                perm_i = torch.randperm(T)
                out[i] = val[i, perm_i, :]
            shuffled[key] = out
        else:
            raise NotImplementedError(f"Time shuffle not implemented for {val.shape}")

    return shuffled


##############################################################################
# 4) Train/Valid Split Along Time
##############################################################################
def split_train_valid(data, train_frac=0.75):
    """
    For each trajectory row, we slice the time dimension:
       - First `train_frac * T` samples => training
       - Remaining => validation
    """
    T = data['t'].shape[1]
    num_train = int(train_frac * T)

    train_data = {}
    valid_data = {}
    for key, val in data.items():
        if val.dim() == 2:  # (num_traj, T)
            train_data[key] = val[:, :num_train]
            valid_data[key] = val[:, num_train:]
        elif val.dim() == 3:  # (num_traj, T, dim)
            train_data[key] = val[:, :num_train, :]
            valid_data[key] = val[:, num_train:, :]
        else:
            raise NotImplementedError()
    return train_data, valid_data


##############################################################################
# 5) Residual NN, Known Physics, and Loss
##############################################################################
class ResidualNN(nn.Module):
    def __init__(self, input_dim, hdim, num_hlayers, output_dim):
        super().__init__()
        layers = [nn.Linear(input_dim, hdim), nn.Tanh()]
        for _ in range(num_hlayers - 1):
            layers += [nn.Linear(hdim, hdim), nn.Tanh()]
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(hdim, output_dim)

    def forward(self, x):
        return self.out(self.hidden(x))


def dynamics_fn(x, t, u, model, M_known, D_known, G_known):
    """
    x: (batch, 6) => [eta, nu]
    u: (batch, 3)
    returns dx = [nu, a_nom + delta]
    """
    eta = x[:, :3]
    nu  = x[:, 3:]
    # known part
    rhs = u - (D_known @ nu.unsqueeze(-1)).squeeze(-1) - (G_known @ eta.unsqueeze(-1)).squeeze(-1)
    batch = x.shape[0]
    M_expanded = M_known.unsqueeze(0).expand(batch, -1, -1)  # (batch, 3, 3)
    a_nom = torch.linalg.solve(M_expanded, rhs.unsqueeze(-1)).squeeze(-1)

    # neural net residual
    # input = concat(x, u) => shape (batch, 9)
    inp = torch.cat([x, u], dim=1)
    delta = model(inp)  # (batch, 3)
    return torch.cat([nu, a_nom + delta], dim=1)


def rk38_step(func, h, x, t, u, model, M_known, D_known, G_known):
    """
    One-step RK38 integrator
    h: (batch,1)
    """
    A = torch.tensor([
        [0,    0,    0,  0],
        [1/3,  0,    0,  0],
        [-1/3, 1,    0,  0],
        [1,   -1,    1,  0]
    ], dtype=x.dtype, device=x.device)
    b = torch.tensor([1/8, 3/8, 3/8, 1/8], dtype=x.dtype, device=x.device)
    c = torch.tensor([0, 1/3, 2/3, 1],     dtype=x.dtype, device=x.device)

    K = []
    for i in range(4):
        if i == 0:
            x_i = x
        else:
            update = torch.zeros_like(x)
            for j in range(i):
                update += h * A[i, j] * K[j]
            x_i = x + update
        t_i = t + h[:, 0]*c[i]
        k_i = func(x_i, t_i, u, model, M_known, D_known, G_known)
        K.append(k_i)
    sumK = torch.zeros_like(x)
    for i in range(4):
        sumK += b[i]*K[i]
    return x + h*sumK


def loss_fn(model, reg_l2, t, x, u, t_next, x_next, M_known, D_known, G_known):
    dt = (t_next - t).unsqueeze(-1)  # (batch,1)
    x_est = rk38_step(dynamics_fn, dt, x, t, u, model, M_known, D_known, G_known)
    mse = torch.mean((x_est - x_next)**2)
    l2 = sum(torch.sum(p**2) for p in model.parameters())
    return mse + reg_l2*l2


##############################################################################
# 6) Training a Single Model with "Update Best"
##############################################################################
def train_single_trajectory(train_data, valid_data, model, optimizer,
                            M_known, D_known, G_known,
                            reg_l2=1e-4,
                            num_epochs=1000):
    """
    For each epoch:
      - compute train loss (with L2 reg)
      - compute val loss (no reg)
      - keep track of the best model state so far (lowest val loss)
    """
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(num_epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        train_loss = loss_fn(model, reg_l2,
                             train_data['t'], train_data['x'], train_data['u'],
                             train_data['t_next'], train_data['x_next'],
                             M_known, D_known, G_known)
        train_loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model, 0.,
                               valid_data['t'], valid_data['x'], valid_data['u'],
                               valid_data['t_next'], valid_data['x_next'],
                               M_known, D_known, G_known)

        # Update best
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

        # Print progress occasionally
        if epoch % 100 == 0:
            print(f" Epoch {epoch:4d} | TrainLoss={train_loss.item():.4f}, ValLoss={val_loss.item():.4f}")

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


##############################################################################
# 7) Train an Ensemble (M Models) with "Best Model" per Trajectory
##############################################################################
def train_ensemble(data_sub, hparams):
    """
    data_sub: dict with shape (M, T-1, ...)
      1) Shuffle time for each trajectory
      2) Split train/valid
      3) Build known physics
      4) For each trajectory i, train single model with best validation
      5) Return list of trained models
    """
    # 1) Shuffle time dimension
    data_shuf = shuffle_time_for_each_trajectory(data_sub, seed=hparams['seed'])

    # 2) Split train/valid
    train_all, valid_all = split_train_valid(data_shuf, train_frac=hparams['train_frac'])

    # 3) Known physics
    dt_vals = data_sub['t_next'] - data_sub['t']
    dt_val = torch.mean(dt_vals).item()
    csad_dp = CSAD_DP_6DOF(dt_val)
    M_known = torch.tensor(six2threeDOF(csad_dp._M), dtype=torch.float32, device=device)
    D_known = torch.tensor(six2threeDOF(csad_dp._D), dtype=torch.float32, device=device)
    G_known = torch.tensor(six2threeDOF(csad_dp._G), dtype=torch.float32, device=device)

    # 4) Train one model per trajectory
    num_traj = data_sub['t'].shape[0]
    ensemble_models = []
    for i in range(num_traj):
        print(f"\n--- Ensemble Model {i}/{num_traj-1} ---")
        traj_train = {k: train_all[k][i] for k in train_all}
        traj_valid = {k: valid_all[k][i] for k in valid_all}

        # Build the residual NN
        model = ResidualNN(
            input_dim=9,   # [6 for x + 3 for u]
            hdim=hparams['hdim'],
            num_hlayers=hparams['num_hlayers'],
            output_dim=3
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'])

        trained_model = train_single_trajectory(
            train_data=traj_train,
            valid_data=traj_valid,
            model=model,
            optimizer=optimizer,
            M_known=M_known,
            D_known=D_known,
            G_known=G_known,
            reg_l2=hparams['regularizer_l2'],
            num_epochs=hparams['num_epochs']
        )
        ensemble_models.append(trained_model)
    return ensemble_models


##############################################################################
# 8) Main: Loop Over Seeds and M-values Exactly Like Spencer's .sh
##############################################################################
def main():
    # If you want to change them, do so here:
    seeds = range(10)
    M_values = [2, 5, 10, 20, 30, 40, 50]

    # You might want to set a global dtype or something:
    # torch.set_default_dtype(torch.float64)  # if desired

    # Hyperparameters
    hparams = {
        'num_hlayers':    2,
        'hdim':           32,
        'train_frac':     0.75,
        'regularizer_l2': 1e-4,
        'learning_rate':  1e-2,
        'num_epochs':     1000,
    }

    # Load the entire dataset once
    raw_data = load_data('training_data_500_40.pkl')
    # shapes: (num_traj, T-1, ...)

    for seed in seeds:
        for M in M_values:
            print(f"\n\n===========================")
            print(f"   SEED = {seed},   M = {M} ")
            print("===========================")

            # 1) Set random seeds
            np.random.seed(seed)
            torch.manual_seed(seed)

            # 2) Sub-sample M trajectories
            data_sub = subsample_trajectories(raw_data, M=M, seed=seed)

            # 3) Update hparams with the seed
            #    (so shuffle_time_for_each_trajectory uses the same seed)
            hparams['seed'] = seed

            # 4) Train the ensemble
            best_ensemble = train_ensemble(data_sub, hparams)

            # 5) Save the ensemble
            out_filename = f"results_seed={seed}_M={M}.pkl"
            with open(out_filename, 'wb') as f:
                pickle.dump(best_ensemble, f)
            print(f"Saved ensemble to {out_filename}!")


if __name__ == "__main__":
    main()
