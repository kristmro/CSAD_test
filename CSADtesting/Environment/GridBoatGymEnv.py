import numpy as np
import gym
from gym import spaces

# --- MCSimPython (Adjust imports to your local structure) ---
from MCSimPython.simulator.csad import CSAD_DP_6DOF
from MCSimPython.waves.wave_loads import WaveLoad
from MCSimPython.waves.wave_spectra import JONSWAP
from MCSimPython.utils import three2sixDOF

try:
    import pygame
except ImportError:
    pygame = None

import time
import matplotlib.pyplot as plt


class GridBoatGymEnv(gym.Env):
    """
    A Gym environment that simulates a boat on a 2D grid with waves & obstacles.
    Perfect for experimenting with model-based RL controllers.
    """

    def __init__(self,
                 dt=0.1,
                 grid_width=15,
                 grid_height=6,
                 render_on=False,
                 final_plot=False):
        super().__init__()

        # Environment parameters
        self.dt = dt
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.render_on = render_on
        self.final_plot = final_plot

        # Initialize simulator objects
        self.vessel = None
        self.waveload = None
        self.simulation_time = 0.0

        # Task definitions
        self.start_position = None
        self.goal = None  # (north, east, size)
        self.obstacles = []  # list of (north, east, size)
        self.wave_conditions = None  # (Hs, Tp, wave_dir_deg)
        self.goal_func = None
        self.obstacle_func = None

        # Rendering
        self.screen = None
        self.clock = None
        self.WINDOW_WIDTH = 750
        self.WINDOW_HEIGHT = 300
        if self.render_on and pygame is not None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            pygame.display.set_caption("Boat Navigation Simulation")
            self.clock = pygame.time.Clock()

        # Scale for rendering
        self.x_scale = self.WINDOW_WIDTH / self.grid_width
        self.y_scale = self.WINDOW_HEIGHT / self.grid_height

        # Action/Observation spaces (example)
        # Action: [Fx, Fy, Mz]
        self.action_space = spaces.Box(
            low=np.array([-2.0, -2.0, -1.0], dtype=np.float32),
            high=np.array([ 2.0,  2.0,  1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )
        # Observation: [boat_n, boat_e, boat_yaw, goal_n, goal_e]
        obs_low = np.array([0.0, 0.0, -np.pi, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array(
            [float(grid_height), float(grid_width), np.pi, float(grid_height), float(grid_width)],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Logging
        self.trajectory = []
        self.previous_action = np.zeros(3)

    def set_task(self, 
                 start_position,
                 goal,
                 wave_conditions,
                 obstacles=None,
                 goal_func=None,
                 obstacle_func=None):
        """
        Define the scenario: 
          - Boat start pose (north, east, heading in degrees)
          - Goal (north, east, size)
          - Wave conditions (Hs, Tp, wave_dir_deg)
          - Obstacles (list of (north, east, size))
          - Optional dynamic goal/obstacle functions
        """
        self.start_position = start_position
        self.goal = goal
        self.wave_conditions = wave_conditions
        self.obstacles = obstacles if obstacles else []
        self.goal_func = goal_func
        self.obstacle_func = obstacle_func
        self.reset()  # Immediately reset with this scenario

    def reset(self):
        """
        Gym API: Reset the environment. 
        Returns the initial observation (np.array).
        """
        # Create a fresh vessel
        self.vessel = CSAD_DP_6DOF(self.dt, method="RK4")
        self.simulation_time = 0.0

        # Set initial pose
        n0, e0, heading_deg = self.start_position
        eta_start = three2sixDOF(np.array([n0, e0, np.deg2rad(heading_deg)]))
        self.vessel.set_eta(eta_start)

        # Waves
        self._set_wave_conditions(*self.wave_conditions)

        # Clear logs
        if self.final_plot:
            self.trajectory = []
        self.previous_action = np.zeros(3)

        # If rendering
        if self.render_on and self.screen is not None:
            self.screen.fill((20, 20, 20))

        # Return first observation
        return self._dict_to_obs(self._get_state())

    def step(self, action):
        """
        Gym API: Step the simulation by one time step with the given action.
        Returns (obs, reward, done, info).
        """
        self.simulation_time += self.dt

        # Dynamic updates to goal or obstacles
        if self.goal_func is not None:
            self.goal = self.goal_func(self.simulation_time)
        if self.obstacle_func is not None:
            self.obstacles = self.obstacle_func(self.simulation_time)

        # Build 6DOF input
        tau_6dof = np.zeros(6)
        tau_6dof[:2] = action[:2]
        tau_6dof[-1] = action[-1]

        # Wave load
        tau_wave = self.waveload(self.simulation_time, self.vessel.get_eta())
        self.vessel.integrate(0, 0, tau_6dof + tau_wave)

        # Check termination
        boat_pos = self.vessel.get_eta()[:2]
        done, info = self._check_termination(boat_pos)

        # Reward (customize!)
        reward = self._compute_reward(action, self.previous_action)

        # Log trajectory if desired
        if self.final_plot:
            self.trajectory.append(boat_pos.copy())

        # Render
        if self.render_on:
            self._render()

        # Prepare next step
        self.previous_action = action

        # Return
        obs = self._dict_to_obs(self._get_state())
        return obs, reward, done, info

    def render(self, mode='human'):
        """
        Optional: For external calls to render if needed.
        """
        if not self.render_on:
            return
        self._render()

    def close(self):
        """
        Clean up if needed.
        """
        if self.render_on and self.screen is not None and pygame is not None:
            pygame.quit()

    # ----------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------

    def _set_wave_conditions(self, hs, tp, wave_dir_deg):
        """
        Creates a wave load from JONSWAP and attaches to the vessel.
        """
        self.wave_conditions = (hs, tp, wave_dir_deg)
        wp = 2 * np.pi / tp
        N_w = 100
        wmin, wmax = wp / 2, 3.0 * wp
        dw = (wmax - wmin) / N_w
        w = np.linspace(wmin, wmax, N_w, endpoint=True)

        jonswap = JONSWAP(w)
        freq, spec = jonswap(hs, tp, gamma=3.3)

        wave_amps = np.sqrt(2 * spec * dw)
        eps = np.random.uniform(0, 2 * np.pi, size=N_w)
        wave_dir = np.ones(N_w) * np.deg2rad(wave_dir_deg)

        self.waveload = WaveLoad(
            wave_amps=wave_amps,
            freqs=w,
            eps=eps,
            angles=wave_dir,
            config_file=self.vessel._config_file,
            interpolate=True,
            qtf_method="geo-mean",
            deep_water=True
        )

    def _get_state(self):
        """
        Return a dictionary describing the boat state, goal, etc.
        """
        eta = self.vessel.get_eta()  # [n, e, z, roll, pitch, yaw]
        return {
            "boat_position": eta[:2],
            "boat_orientation": eta[5],
            "goal": self.goal,
            "obstacles": self.obstacles,
        }

    def _dict_to_obs(self, state_dict):
        """
        Convert dict to numeric observation vector: 
        e.g. [boat_n, boat_e, boat_yaw, goal_n, goal_e].
        """
        boat_n, boat_e = state_dict["boat_position"]
        boat_yaw = state_dict["boat_orientation"]
        g_n, g_e, _ = state_dict["goal"]
        return np.array([boat_n, boat_e, boat_yaw, g_n, g_e], dtype=np.float32)

    def _check_termination(self, boat_pos):
        """
        Check if the boat has reached the goal or collided with an obstacle.
        """
        goal_n, goal_e, goal_size = self.goal
        # Goal check
        if (goal_n - goal_size / 2 <= boat_pos[0] <= goal_n + goal_size / 2 and
            goal_e - goal_size / 2 <= boat_pos[1] <= goal_e + goal_size / 2):
            return True, {"reason": "goal_reached"}

        # Obstacle check
        for obs_n, obs_e, obs_size in self.obstacles:
            dist = np.linalg.norm(boat_pos - np.array([obs_n, obs_e]))
            if dist < obs_size:
                return True, {"reason": "collision"}

        return False, {}

    def _compute_reward(self, action, prev_action):
        """
        A placeholder reward function (0.0). 
        Modify for your MBRL approach:
          - distance to goal
          - collision penalty
          - heading alignment
          - control cost, etc.
        """
        return 0.0

    def _render(self):
        """
        Internal rendering with Pygame. 
        """
        if self.screen is None or pygame is None:
            return

        self.screen.fill((20, 20, 20))
        self._draw_grid()
        self._draw_goal()
        self._draw_obstacles()
        self._draw_boat()
        pygame.display.flip()
        self.clock.tick(60)

    def _draw_grid(self):
        import pygame
        grid_color = (50, 50, 50)
        for x in range(self.grid_width + 1):
            start_px = (x * self.x_scale, 0)
            end_px = (x * self.x_scale, self.WINDOW_HEIGHT)
            pygame.draw.line(self.screen, grid_color, start_px, end_px, 1)
        for y in range(self.grid_height + 1):
            start_px = (0, y * self.y_scale)
            end_px = (self.WINDOW_WIDTH, y * self.y_scale)
            pygame.draw.line(self.screen, grid_color, start_px, end_px, 1)

    def _draw_goal(self):
        import pygame
        g_n, g_e, g_size = self.goal
        px = g_e * self.x_scale
        py = (self.grid_height - g_n) * self.y_scale
        radius_px = (g_size / 2) * self.x_scale
        pygame.draw.circle(self.screen, (255, 215, 0), (int(px), int(py)), int(radius_px))

    def _draw_obstacles(self):
        import pygame
        for obs_n, obs_e, obs_size in self.obstacles:
            px = obs_e * self.x_scale
            py = (self.grid_height - obs_n) * self.y_scale
            radius_px = (obs_size / 2) * self.x_scale
            pygame.draw.circle(self.screen, (200, 0, 0), (int(px), int(py)), int(radius_px))

    def _draw_boat(self):
        import pygame
        eta = self.vessel.get_eta()
        boat_n, boat_e, boat_yaw = eta[0], eta[1], eta[5]

        # Build local hull shape (rectangle or something else).
        # This is just a minimal shape example:
        hull_pts_local = np.array([
            [-0.5,  0.3],
            [ 0.5,  0.3],
            [ 0.5, -0.3],
            [-0.5, -0.3],
        ])

        # Rotate by boat_yaw
        c = np.cos(boat_yaw)
        s = np.sin(boat_yaw)
        rot = np.array([[c, s], [-s, c]])

        pixel_pts = []
        for (lx, ly) in hull_pts_local:
            gx, gy = rot @ np.array([lx, ly])
            # Shift
            gx += boat_e
            gy += boat_n
            # Convert to screen coords
            sx = int(gx * self.x_scale)
            sy = int((self.grid_height - gy) * self.y_scale)
            pixel_pts.append((sx, sy))

        pygame.draw.polygon(self.screen, (0, 100, 255), pixel_pts)

    def plot_trajectory(self):
        """
        If final_plot=True, you can call this after an episode to see the path.
        """
        if not self.final_plot or len(self.trajectory) == 0:
            return
        traj = np.array(self.trajectory)
        plt.figure(figsize=(8, 4))
        plt.plot(traj[:, 1], traj[:, 0], 'g-', label="Boat Trajectory")
        g_n, g_e, g_s = self.goal
        plt.scatter(g_e, g_n, c='yellow', s=(g_s * self.x_scale)**2, edgecolor='black', label="Goal")
        for obs_n, obs_e, obs_size in self.obstacles:
            plt.scatter(obs_e, obs_n, c='red', s=(obs_size * self.x_scale)**2, edgecolor='black', label="Obstacle")

        plt.xlim([0, self.grid_width])
        plt.ylim([0, self.grid_height])
        plt.xlabel("East [m]")
        plt.ylabel("North [m]")
        plt.title(f"Boat Trajectory in {self.grid_width}×{self.grid_height} Domain")
        plt.legend(loc='upper right', fontsize='small', scatterpoints=1, markerscale=0.1)
        plt.grid(True)
        plt.show()


# ------------------------------------------------------------------------
# EXAMPLE USAGE with a placeholder Model-Based RL "controller"
# ------------------------------------------------------------------------

def mbrl_controller(observation):
    """
    Placeholder for your model-based RL planner / policy.
    observation = [boat_n, boat_e, boat_yaw, goal_n, goal_e]
    Return an action = [Fx, Fy, Mz].
    For now, just return a zero or random control, etc.
    """
    # Example: random (replace with real MBRL planning code)
    return np.random.uniform(low=[-1, -1, -0.5], high=[1, 1, 0.5])


def main():
    """
    Quick demo running the environment with a placeholder MBRL controller.
    """
    env = GridBoatGymEnv(dt=0.1, grid_width=15, grid_height=6, render_on=True, final_plot=True)

    # Example task
    start_pos = (2, 2, 90)              # north=2, east=2, heading=90 deg
    goal = (4, 12, 1.0)                 # north=4, east=12, size=1.0
    wave_conditions = (1.0, 4.5, 0)     # (Hs=1.0, Tp=4.5, wave_dir=0 deg)
    obstacles = [(2, 7, 1.0)]           # single obstacle at (n=2, e=7, size=1)

    env.set_task(
        start_position=start_pos,
        goal=goal,
        wave_conditions=wave_conditions,
        obstacles=obstacles,
        goal_func=None,      # or define a function to move the goal
        obstacle_func=None   # or define a function to move obstacles
    )

    obs = env.reset()
    done = False
    step_count = 0
    max_steps = 200

    while not done and step_count < max_steps:
        action = mbrl_controller(obs)
        obs, reward, done, info = env.step(action)
        step_count += 1
        # For debugging
        # print(f"Step={step_count}, Obs={obs}, Reward={reward}, Done={done}, Info={info}")

    env.plot_trajectory()
    env.close()

    if done:
        print("Episode finished due to:", info.get("reason", "unknown"))
    else:
        print("Max steps reached.")


if __name__ == "__main__":
    main()
