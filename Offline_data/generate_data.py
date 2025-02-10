import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# -----------------------------------------------------
# 1. Define the Corridor (Path) Reward Functions
# -----------------------------------------------------
def corridor_a(x, y):
    """
    Corridor a: Horizontal path from (2,2) to (4,2)
    Ignores deviation in x (multiplied by 0) and penalizes deviation in y.
    """
    return np.where((x >= 2) & (x <= 4), -(y - 2)**2, -100)

def corridor_b(x, y):
    """
    Corridor b: Vertical path from (2,2) to (2,4)
    Ignores deviation in y and penalizes deviation in x.
    """
    return np.where((y >= 2) & (y <= 4), -(x - 2)**2, -100)

def corridor_c(x, y):
    """
    Corridor c: Horizontal path from (2,4) to (4,4)
    Ignores deviation in x and penalizes deviation in y.
    """
    return np.where((x >= 2) & (x <= 4), -(y - 4)**2, -100)

def corridor_d(x, y):
    """
    Corridor d: Vertical path from (4,2) to (4,4)
    Ignores deviation in y and penalizes deviation in x.
    """
    return np.where((y >= 2) & (y <= 4), -(x - 4)**2, -100)

def corridor_reward(x, y):
    """
    Computes the corridor reward by taking the best (maximum) reward
    among corridors a, b, c, and d.
    """
    r_a = corridor_a(x, y)
    r_b = corridor_b(x, y)
    r_c = corridor_c(x, y)
    r_d = corridor_d(x, y)
    # Take the maximum reward (remember: rewards are negative, so "maximum" means closer to zero).
    return np.maximum(np.maximum(r_a, r_b), np.maximum(r_c, r_d))

# -----------------------------------------------------
# 2. Define the Bonus Function for Waypoints
# -----------------------------------------------------
def bonus_reward(x, y, waypoints, bonus_scale=10, sigma=0.2):
    """
    Adds a Gaussian bonus at each waypoint.
    
    bonus = bonus_scale * exp(-((x - wx)^2 + (y - wy)^2) / (2*sigma^2))
    
    Parameters:
      waypoints: list of (wx, wy) tuples.
      bonus_scale: magnitude of the bonus.
      sigma: spread of the Gaussian bump.
    """
    # Ensure bonus is a float array to avoid casting issues.
    bonus = np.zeros_like(x, dtype=np.float64)
    for (wx, wy) in waypoints:
        bonus += bonus_scale * np.exp(-(((x - wx)**2 + (y - wy)**2) / (2 * sigma**2)))
    return bonus

# -----------------------------------------------------
# 3. Define the Composite Reward Function
# -----------------------------------------------------
def composite_reward(x, y, waypoints, bonus_scale=10, sigma=0.2):
    """
    Composite reward = corridor reward (the best along the path) + waypoint bonus.
    """
    return corridor_reward(x, y) + bonus_reward(x, y, waypoints, bonus_scale, sigma)

# -----------------------------------------------------
# 4. Set Up the Grid and Compute the Reward
# -----------------------------------------------------
# Define a grid that covers the region of interest.
x_vals = np.linspace(0, 6, 300)
y_vals = np.linspace(0, 6, 300)
X, Y = np.meshgrid(x_vals, y_vals)

# Define the waypoints (points of interest along the path)
# These are the endpoints: (2,2), (4,2), (4,4), and (2,4)
waypoints = [(2, 2), (4, 2), (4, 4), (2, 4)]

# Compute the composite reward over the grid.
R_total = composite_reward(X, Y, waypoints, bonus_scale=10, sigma=0.8)

# -----------------------------------------------------
# 5. Plot the Composite Reward Function in 3D
# -----------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, R_total, cmap='viridis', edgecolor='none', alpha=0.9)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Reward')
ax.set_title('Composite Reward: Path Corridors + Waypoint Bonuses')
fig.colorbar(surf, shrink=0.5, aspect=5)

# Mark the waypoints on the plot.
for i, (wx, wy) in enumerate(waypoints):
    # Compute the reward at the waypoint (should be high due to the bonus).
    r_wp = composite_reward(np.array([wx]), np.array([wy]), waypoints, bonus_scale=10, sigma=0.2).item()
    # Label the first waypoint; others will share the label.
    ax.scatter(wx, wy, r_wp, color='red', s=80, marker='o', label='Waypoint' if i == 0 else "")
ax.legend(loc='upper right')

plt.show()
