#!/usr/bin/env python3
"""
DiffUtils.py

Differentiable utility functions for marine simulations using PyTorch.
Contains functions like Rz_torch, diff_J, pipi, to_positive_angle, and
differentiable_interp1d with an extra "dim" parameter, as well as three2sixDOF
and six2threeDOF conversions.
"""

import torch
import math

def Rx_torch(phi):
    """3x3 rotation matrix about x-axis."""
    c = torch.cos(phi)
    s = torch.sin(phi)
    return torch.stack([
        torch.stack([torch.tensor(1.0, dtype=phi.dtype, device=phi.device), torch.tensor(0.0, dtype=phi.dtype, device=phi.device), torch.tensor(0.0, dtype=phi.dtype, device=phi.device)]),
        torch.stack([torch.tensor(0.0, dtype=phi.dtype, device=phi.device),              c,                          -s]),
        torch.stack([torch.tensor(0.0, dtype=phi.dtype, device=phi.device),              s,                           c])
    ])

def Ry_torch(theta):
    """3x3 rotation matrix about y-axis."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.stack([
        torch.stack([     c, torch.tensor(0.0, dtype=theta.dtype, device=theta.device),     s]),
        torch.stack([torch.tensor(0.0, dtype=theta.dtype, device=theta.device), torch.tensor(1.0, dtype=theta.dtype, device=theta.device), torch.tensor(0.0, dtype=theta.dtype, device=theta.device)]),
        torch.stack([    -s, torch.tensor(0.0, dtype=theta.dtype, device=theta.device),     c])
    ])

def Rz_torch(psi):
    """3x3 rotation matrix about z-axis."""
    c = torch.cos(psi)
    s = torch.sin(psi)
    return torch.stack([
        torch.stack([     c, -s, torch.tensor(0.0, dtype=psi.dtype, device=psi.device)]),
        torch.stack([     s,  c, torch.tensor(0.0, dtype=psi.dtype, device=psi.device)]),
        torch.stack([torch.tensor(0.0, dtype=psi.dtype, device=psi.device), torch.tensor(0.0, dtype=psi.dtype, device=psi.device), torch.tensor(1.0, dtype=psi.dtype, device=psi.device)])
    ])

def Rzyx_torch(phi, theta, psi):
    """
    Composite rotation matrix: Rz(psi)*Ry(theta)*Rx(phi), matching 'Rzyx' in old code.
    We assume order phi->theta->psi means Rz(psi) * Ry(theta) * Rx(phi).
    """
    return Rz_torch(psi) @ Ry_torch(theta) @ Rx_torch(phi)

def Tzyx_torch(phi, theta, psi):
    """
    3x3 Euler angle rate transform matrix matching 'Tzyx(eta)' in old code:
       T( phi, theta ) = 
         [[1,  sin(phi)*tan(theta),  cos(phi)*tan(theta)],
          [0,  cos(phi),            -sin(phi)],
          [0,  sin(phi)/cos(theta),  cos(phi)/cos(theta)]]
    The 'psi' does not appear in the expressions, 
    but we keep the same signature for consistency.
    """
    sinp = torch.sin(phi)
    cosp = torch.cos(phi)
    sint = torch.sin(theta)
    cost = torch.cos(theta)
    tant = torch.tan(theta)

    return torch.stack([
        torch.stack([torch.tensor(1.0, dtype=phi.dtype, device=phi.device),  sinp*tant,   cosp*tant]),
        torch.stack([torch.tensor(0.0, dtype=phi.dtype, device=phi.device),      cosp,         -sinp]),
        torch.stack([torch.tensor(0.0, dtype=phi.dtype, device=phi.device),  sinp/cost,    cosp/cost])
    ])

def J_torch(eta):
    """
    6x6 transform matrix matching the old 'J(eta)' function:
      J(eta) = [[Rzyx(eta[3],eta[4],eta[5]),  0],
                [0,                           Tzyx(eta[3],eta[4],eta[5])]]
    where eta is [x, y, z, roll, pitch, yaw].
    """
    phi   = eta[3]
    theta = eta[4]
    psi   = eta[5]

    R_part = Rzyx_torch(phi, theta, psi)          # shape (3,3)
    T_part = Tzyx_torch(phi, theta, psi)          # shape (3,3)

    # Build top row block [ R , 0 ]
    top = torch.cat([
        R_part, 
        torch.zeros((3,3), dtype=eta.dtype, device=eta.device)
    ], dim=1)  # shape (3,6)

    # Build bottom row block [ 0 , T ]
    bottom = torch.cat([
        torch.zeros((3,3), dtype=eta.dtype, device=eta.device),
        T_part
    ], dim=1)  # shape (3,6)

    # Full 6x6
    return torch.cat([top, bottom], dim=0)


# --------------------------------------------------------------------------
# Minimal 1D interpolation function in PyTorch to replace scipy's interp1d.
# This function is a simple approach to do linear interpolation along one axis,
# with optional boundary fill.
# --------------------------------------------------------------------------
def torch_lininterp_1d(x, y, x_new, axis=0, left_fill=None, right_fill=None):
    """
    A minimal linear interpolation in PyTorch for 1D data, 
    replicating 'bounds_error=False, fill_value=(left_fill, right_fill)':

    - x: 1D sorted Tensor of shape (N,).
    - y: Tensor with shape (N, ...) if axis=0, or (..., N) if axis=-1, or something else.
    - x_new: 1D or ND Tensor of new points at which to interpolate.
    - axis: the axis in y that corresponds to x's dimension. By default axis=0.
    - left_fill, right_fill: either None or scalar/broadcastable
                             to fill out-of-bounds queries. If None => we use y's boundary.

    Returns: a Tensor shaped like y but with dimension 'axis' replaced by x_new's shape.
    """
    if x.dim() != 1:
        raise ValueError("x must be 1D and sorted.")
    if axis != 0:
        # We'll permute axis to the front
        dims = list(range(y.dim()))
        if axis < 0:
            axis = y.dim() + axis
        perm = [axis] + dims[:axis] + dims[axis+1:]
        y_perm = y.permute(perm)  # shape (N, *rest)
    else:
        y_perm = y

    N = x.shape[0]
    shape_rest = y_perm.shape[1:]
    M = 1
    for s in shape_rest:
        M *= s

    # Flatten the "rest" so we have (N, M)
    y_flat = y_perm.reshape(N, M)
    x_new_flat = x_new.flatten()
    K = x_new_flat.shape[0]

    # Bucketize to find intervals
    idx = torch.bucketize(x_new_flat, x)
    # clamp to [1, N-1]
    idx = torch.clamp(idx, 1, N-1)

    # Gather x0, x1
    x0 = x[idx - 1]
    x1 = x[idx]

    denom = (x1 - x0).clone()
    denom[denom == 0] = 1e-9  # avoid div-by-zero

    w = (x_new_flat - x0) / denom  # shape(K,)

    # We'll build out by gathering from y_flat
    out_flat = torch.empty((K, M), dtype=y.dtype, device=y.device)
    for col in range(M):
        col_y = y_flat[:, col]     # shape (N,)
        y0_col = col_y[idx - 1]    # lower bin edges
        y1_col = col_y[idx]
        interped = y0_col + (y1_col - y0_col) * w
        out_flat[:, col] = interped

    # Apply out-of-bounds fill
    left_mask = x_new_flat < x[0]
    right_mask = x_new_flat > x[-1]
    if left_fill is not None:
        out_flat[left_mask] = left_fill
    else:
        # if None => use y_flat[0,:]
        for col in range(M):
            out_flat[left_mask, col] = y_flat[0, col]

    if right_fill is not None:
        out_flat[right_mask] = right_fill
    else:
        # if None => use y_flat[-1,:]
        for col in range(M):
            out_flat[right_mask, col] = y_flat[-1, col]

    # Reshape => x_new shape + shape_rest
    out_shape = list(x_new.shape) + list(shape_rest)
    out_reshaped = out_flat.view(*out_shape)

    # if we permuted, un-permute:
    if axis != 0:
        inv_perm = [0]*len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out_final = out_reshaped.permute(inv_perm)
    else:
        out_final = out_reshaped

    return out_final


# --------------------------------------------------------------------------
# Utility angle functions
# --------------------------------------------------------------------------
def pipi(theta):
    """Return angle mapped to [-pi, pi)."""
    # For differentiability, we can do a modulo-like approach, 
    # but note it might create discontinuities in the gradient
    return torch.remainder(theta + math.pi, 2*math.pi) - math.pi

def to_positive_angle(theta):
    """
    Map angle from [-pi, pi) to [0, 2*pi).
    Caution: This is piecewise, can break gradients at boundary.
    """
    return torch.where(theta < 0, theta + 2*math.pi, theta)

def three2sixDOF(v):
    """
    Convert a 3DOF vector or matrix to 6DOF.
    If v is a vector of shape (3,), returns tensor of shape (6,) with mapping:
      [v0, v1, 0, 0, 0, v2].
    If v is a matrix of shape (3,3), flatten and embed into a 6x6 matrix.
    """
    if v.dim() == 1:
        return torch.tensor([v[0], v[1], 0.0, 0.0, 0.0, v[2]], dtype=v.dtype, device=v.device)
    elif v.dim() == 2:
        flat = v.flatten()
        part1 = flat[0:2]
        part2 = torch.zeros(3, dtype=v.dtype, device=v.device)
        part3 = flat[2:5]
        part4 = torch.zeros(3, dtype=v.dtype, device=v.device)
        part5 = flat[5:6]
        part6 = torch.zeros(18, dtype=v.dtype, device=v.device)
        part7 = flat[6:8]
        part8 = torch.zeros(3, dtype=v.dtype, device=v.device)
        part9 = flat[8:9]
        out_flat = torch.cat([part1, part2, part3, part4, part5, part6, part7, part8, part9])
        return out_flat.view(6,6)
    else:
        raise ValueError("Input v must be 1D or 2D tensor.")

def six2threeDOF(v):
    """
    Convert a 6DOF vector or matrix to 3DOF.
    If v is a vector of shape (6,), returns tensor with elements [0, 1, 5].
    If v is a matrix of shape (6,6), returns the submatrix at rows and columns [0,1,5].
    """
    if v.dim() == 1:
        return v[[0,1,5]]
    elif v.dim() == 2:
        return v[[0,1,5]][:,[0,1,5]]
    else:
        raise ValueError("Input v must be 1D or 2D tensor.")
