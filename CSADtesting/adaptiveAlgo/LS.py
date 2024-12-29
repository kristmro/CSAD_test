import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.linalg

# Given constants (true values for the pendulum)
m_true = 0.2     # Mass in kg
l_true = 0.4      # Length in m
beta_true = 0.003 # Damping coefficient in kg·m²/s
g = 9.81          # Gravitational acceleration in m/s²

# True theta values for comparison
theta_true = np.array([m_true * l_true**2, beta_true, m_true * l_true])

# Desired energy at the upright position
E_d = 2 * m_true * g * l_true

# Adaptive update law functions
alpha = 0  # Normalization factor (set to zero for pure LS)
beta = 0.2   # Forgetting factor (set to zero for pure LS)

def covdot(phi, beta, P):
    m = np.sqrt(abs(1 + alpha * (phi.T @ phi)))
    return beta * P - P @ (phi @ phi.T / m**2) @ P

def thetadot(theta, z, phi, P):
    m = np.sqrt(abs(1 + alpha * (phi.T @ phi)))
    epsilon = (z - theta.T @ phi) / m**2
    theta_update = P @ phi * epsilon
    return theta_update

# Swing-up control using adaptive parameters
def adaptive_swing_up_control(q, q_dot, theta_hat, g=9.81, k=0.1):
    m_hat_l_squared, _, m_hat_l = theta_hat
    if m_hat_l <= 0:
        m_hat_l = 0.001
    if m_hat_l_squared <= 0:
        m_hat_l_squared = 0.001
    E = 0.5 * m_hat_l_squared * q_dot**2 + m_hat_l * g * (1 - np.cos(q))
    E_tilde = E - E_d
    tau_control = -k * E_tilde * np.sign(q_dot)
    tau_control = np.clip(tau_control, -0.5, 0.5)  # Torque saturation
    return tau_control

# Stabilizing controller using adaptive parameters (LQR)
def adaptive_stabilizing_lqr_control(q, q_dot, theta_hat, g=9.81):
    m_hat_l_squared, beta_hat, m_hat_l = theta_hat
    if m_hat_l <= 0:
        m_hat_l = 0.001
    if m_hat_l_squared <= 0:
        m_hat_l_squared = 0.001
    if beta_hat <= 0:
        beta_hat = 0.001
    A = np.array([[0, 1], [g / m_hat_l, -beta_hat / m_hat_l_squared]])
    B = np.array([[0], [1 / m_hat_l_squared]])
    Q = np.diag([10, 1])
    R = np.array([[0.1]])
    try:
        P_lqr = scipy.linalg.solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P_lqr
    except:
        K = np.array([[0, 0]])
    x = np.array([q - np.pi, q_dot]).reshape(-1, 1)
    tau_control = (-K @ x).item()
    tau_control = np.clip(tau_control, -0.5, 0.5)  # Torque saturation
    return tau_control

# Combined control with parameter estimation
def combined_adaptive_control(t, y):
    # Extract variables from y
    q = y[0]
    q_dot = y[1]
    theta = y[2:5].reshape(-1, 1)
    P_flat = y[5:]
    P = P_flat.reshape(3, 3)

    # Use theta_hat for control
    theta_hat = theta.flatten()

    # Determine control law based on current state
    if abs(q - np.pi) < 1 and abs(q_dot) < 1:
        tau_control = adaptive_stabilizing_lqr_control(q, q_dot, theta_hat)
    else:
        tau_control = adaptive_swing_up_control(q, q_dot, theta_hat)

    # Compute q_ddot using true parameters
    q_ddot = (tau_control - beta_true * q_dot - m_true * g * l_true * np.sin(q)) / (m_true * l_true**2)

    # Form phi and z
    phi = np.array([q_ddot, q_dot, g * np.sin(q)]).reshape(-1, 1)
    z = tau_control

    # Compute P_dot and theta_dot
    P_dot = covdot(phi, beta, P)
    theta_dot = thetadot(theta, z, phi, P)

    # Flatten P_dot for ODE integration
    P_dot_flat = P_dot.flatten()

    # Construct the derivative of the state vector
    dydt = np.zeros_like(y)
    dydt[0] = q_dot
    dydt[1] = q_ddot
    dydt[2:5] = theta_dot.flatten()
    dydt[5:] = P_dot_flat

    return dydt

# Initial conditions
q_initial = 0.2     # Initial angle (rad)
q_dot_initial = 0.1 # Initial angular velocity (rad/s)
theta_initial = np.array([0.0, 0.000, 0.0])  # Initial parameter estimates
P_initial = np.linalg.inv(np.diag([0.13, 10, 0.06]))  # Initial covariance matrix

# Flatten initial P for the ODE solver
P_initial_flat = P_initial.flatten()

# Combine all initial conditions into a single vector
y0 = np.concatenate(([q_initial, q_dot_initial], theta_initial, P_initial_flat))

# Time vector for simulation
time_vals = np.linspace(0, 30, 200)

# Solve the system with combined adaptive control
sol = solve_ivp(combined_adaptive_control, [time_vals[0], time_vals[-1]], y0, t_eval=time_vals, method='RK45')

# Extract the results
q = sol.y[0]
q_dot = sol.y[1]
theta_estimates = sol.y[2:5]  # Estimated parameters over time

# Plot the angle and angular velocity over time
plt.figure(figsize=(12, 6))
plt.plot(time_vals, q, label='Angle (q) [rad]')
plt.plot(time_vals, q_dot, label='Angular velocity (q_dot) [rad/s]')
plt.axhline(np.pi, color='gray', linestyle='--', label='Upright Position (q=π)')
plt.xlabel('Time (s)')
plt.ylabel('State values')
plt.title('Response of Nonlinear Pendulum with Combined Adaptive Control (LS)')
plt.legend()
plt.grid()
plt.show()

# Plot the estimated parameters vs. true values
parameter_names = [r'$m l^2$', r'$\beta$', r'$m l$']
true_values = theta_true

plt.figure(figsize=(12, 6))
for i in range(3):
    plt.plot(time_vals, theta_estimates[i, :], label=f'Estimated {parameter_names[i]}')
    plt.axhline(true_values[i], color='gray', linestyle='--', label=f'True {parameter_names[i]}' if i == 0 else "")
plt.xlabel('Time (s)')
plt.ylabel('Parameter values')
plt.title('Estimated Parameters with LS Estimation (Using sin(q))')
plt.legend()
plt.grid(True)
plt.show()
