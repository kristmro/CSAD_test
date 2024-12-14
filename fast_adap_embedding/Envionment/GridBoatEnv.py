import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from MCSimPython.simulator.csad import CSAD_DP_6DOF
from MCSimPython.waves.wave_loads import WaveLoad
from MCSimPython.waves.wave_spectra import JONSWAP
from MCSimPython.utils import three2sixDOF


class GridWaveEnvironment:
    """Grid-based Wave Environment with real-time rendering."""

    def __init__(self, dt=0.1, grid_size=100, render_on=False, final_plot=False):
        """
        Initialize the environment.

        Args:
            dt (float): Simulation time step.
            grid_size (int): Size of the grid (default: 100x100).
            render_on (bool): Whether to enable rendering during the simulation.
            final_plot (bool): Whether to display a final trajectory plot at the end.
        """
        self.dt = dt
        self.grid_size = grid_size
        self.vessel = CSAD_DP_6DOF(dt)
        self.waveload = None
        self.simulation_time = 0.0  # Initialize simulation time
        self.render_on = render_on
        self.final_plot = final_plot
        self.fig, self.ax = None, None
        self.start_position = None
        self.goal = None
        self.obstacles = []
        self.wave_conditions = None
        self.trajectory = []  # Store trajectory for final plotting

    def set_task(self, start_position, goal, wave_conditions, obstacles=None):
        """
        Configure a task for the environment.

        Args:
            start_position (tuple): Boat's initial (x, y, yaw in degrees).
            goal (tuple): Goal defined as (x, y, size).
            wave_conditions (tuple): Wave conditions (Hs, Tp, wave_direction_deg).
            obstacles (list): List of obstacles [(x, y, size), ...].
        """
        self.start_position = start_position
        self.goal = goal
        self.wave_conditions = wave_conditions
        self.obstacles = obstacles if obstacles else []
        self.reset()

    def set_wave_conditions(self, hs, tp, wave_dir_deg):
        """Set wave conditions for the environment."""
        self.wave_conditions = (hs, tp, wave_dir_deg)
        wp = 2 * np.pi / tp
        N_w = 100  # Number of wave components
        wmin, wmax = wp / 2, 3. * wp
        dw = (wmax - wmin) / N_w
        w = np.linspace(wmin, wmax, N_w, endpoint=True)

        jonswap = JONSWAP(w)
        freq, spec = jonswap(hs, tp, gamma=3.3)

        wave_amps = np.sqrt(2 * spec * dw)
        eps = np.random.uniform(0, 2 * np.pi, size=N_w)
        wave_dir = np.ones(N_w) * np.deg2rad(wave_dir_deg)

        self.waveload = WaveLoad(
            wave_amps=wave_amps, freqs=w, eps=eps, angles=wave_dir,
            config_file=self.vessel._config_file, interpolate=True, qtf_method="geo-mean", deep_water=True
        )

    def step(self, action):
        """
        Simulate a single time step.

        Args:
            action (np.ndarray): Action array [Fx, Fy, Mz].

        Returns:
            tuple: (state, reward, done, info)
        """
        self.simulation_time += self.dt  # Increment simulation time
        #print(f"Simulation Time: {self.simulation_time}")
        # Forces and moments
        tau_6dof = np.zeros(6)
        tau_6dof[:2] = action[:2]
        tau_6dof[-1] = action[-1]

        # Calculate wave forces
        #tau_w = np.zeros(6)
        tau_wave = self.waveload(self.simulation_time, self.vessel.get_eta())
        print(f"Wave Forces: {tau_wave[0]}")
        #tau_ws = tau_w + tau_sv
        #print(f"Wave Forces: {tau_wave}")
        # Integrate vessel dynamics
        self.vessel.integrate(0, 0, tau_6dof + tau_wave) #zero current and beta_c


        # Check for goal and obstacles
        boat_pos = self.vessel.get_eta()[:2]
        #print(f"Boat Position: {self.vessel.get_eta()}")
        #print(f"Boat speed: {self.vessel.get_nu()}")


        #print(self.vessel.get_eta()[2])
        #print(f"Boat Position: {boat_pos}")
        done, info = self._check_termination(boat_pos)

        # Compute reward
        #reward = self.compute_reward(action, self.previous_action)
        #self.previous_action = action
        reward = 0

        # Append trajectory for final plotting
        if self.final_plot:
            self.trajectory.append(self.vessel.get_eta()[:2])

        # Render if enabled
        if self.render_on:
            self.render()

        return self.get_state(), done, info, reward

    def reset(self):
        """Reset the vessel and environment."""
        self.vessel = CSAD_DP_6DOF(self.dt, method="RK4") # MUST HAVE RK4!!
        self.simulation_time = 0.0
        eta_start = three2sixDOF(np.array([self.start_position[0], self.start_position[1], np.deg2rad(self.start_position[-1])]))
        self.vessel.set_eta(eta_start)
        self.set_wave_conditions(*self.wave_conditions) # 2.5,12.5,0->self,hs,tp,dir
        self.previous_action = np.zeros(3)

        if self.final_plot:
            self.trajectory = []

        if self.render_on:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlim(0, self.grid_size)
            self.ax.set_ylim(0, self.grid_size)
            self.ax.set_title("Boat Navigation Simulation")
    def get_state(self):
        """
        Get the current state of the environment.

        Returns:
            dict: Contains boat position, orientation, goal, obstacles, wave conditions, and trajectory.
        """
        eta = self.vessel.get_eta()  # 6DOF state of the vessel
        state = {
            "boat_position": eta[:2],  # [North, East]
            "boat_orientation": eta[-1],  # Yaw (heading angle in radians)
            "goal": self.goal,  # Goal position and size
            "obstacles": self.obstacles,  # Obstacles as list of (x, y, size)
            "wave_conditions": self.wave_conditions,  # Wave parameters (Hs, Tp, direction)
        }
        return state
    def _check_termination(self, boat_pos):
        """
        Check if the boat has reached the goal or collided with obstacles.

        Args:
            boat_pos (np.ndarray): Boat's current position [North, East].

        Returns:
            tuple: (done, info) - where `done` is a boolean indicating if the episode is over,
                and `info` provides additional context (e.g., 'goal_reached' or 'collision').
        """
        # Check if boat is within the goal area
        goal_x, goal_y, goal_size = self.goal
        if (goal_x - goal_size / 2 <= boat_pos[1] <= goal_x + goal_size / 2 and
            goal_y - goal_size / 2 <= boat_pos[0] <= goal_y + goal_size / 2):
            return True, {"reason": "goal_reached"}

        # Check for collision with obstacles
        for obs_x, obs_y, obs_size in self.obstacles:
            if np.linalg.norm(boat_pos - np.array([obs_y, obs_x])) < obs_size:
                return True, {"reason": "collision"}

        # If neither condition is met, the episode continues
        return False, {}


    def render(self):
        """Render the current state of the environment."""
        self.ax.clear()
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)

        # Draw goal
        goal_x, goal_y, goal_size = self.goal
        self.ax.add_patch(plt.Rectangle((goal_x - goal_size / 2, goal_y - goal_size / 2),
                                        goal_size, goal_size, color='yellow', alpha=0.6))

        # Draw obstacles
        for obs_x, obs_y, obs_size in self.obstacles:
            self.ax.add_patch(plt.Rectangle((obs_x - obs_size / 2, obs_y - obs_size / 2),
                                            obs_size, obs_size, color='red', alpha=0.6))

        # Draw boat
        boat_pos = self.vessel.get_eta()[:2]
        boat_yaw = self.vessel.get_eta()[-1]
        boat_length, boat_width = 2.578 * 3, 0.3 * 3 # Length: 2.578 m, Width: 0.3 m (scaled by 3)
        rotation_matrix = np.array([
            [np.cos(boat_yaw), -np.sin(boat_yaw)],
            [np.sin(boat_yaw), np.cos(boat_yaw)]
        ])
        boat_corners = np.array([
            [-boat_length / 2, -boat_width / 2],
            [boat_length / 2, -boat_width / 2],
            [boat_length / 2, boat_width / 2],
            [-boat_length / 2, boat_width / 2]
        ]).T
        rotated_corners = (rotation_matrix @ boat_corners).T + boat_pos[:2]
        self.ax.add_patch(plt.Polygon(rotated_corners, color='blue', alpha=0.8))
        arrow_length = boat_length * 0.53
        arrow_pos = np.mean(rotated_corners[:2], axis=0) 
        self.ax.arrow(arrow_pos[0], arrow_pos[1],
                      arrow_length * np.cos(boat_yaw),
                      arrow_length * np.sin(boat_yaw),
                      head_width=0.3, head_length=0.6, fc='black', ec='black') # Arrow for heading
        plt.pause(0.01)

    def plot_trajectory(self):
        """Plot the trajectory after the simulation."""
        if self.final_plot:
            trajectory = np.array(self.trajectory)
            plt.figure(figsize=(8, 8))
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label="Trajectory")
            plt.scatter(self.goal[0], self.goal[1], c='yellow', label="Goal", s=100, edgecolor='black')
            for obs_x, obs_y, obs_size in self.obstacles:
                plt.gca().add_patch(plt.Circle((obs_x, obs_y), obs_size, color='red', alpha=0.5, label="Obstacle"))
            plt.xlim(0, self.grid_size)
            plt.ylim(0, self.grid_size)
            plt.xlabel("E [m]")
            plt.ylabel("N [m]")
            plt.legend()
            plt.title("Final Trajectory")
            plt.show()


    def compute_reward(self, action, previous_action):
        """
        Compute the reward based on the current state and action.

        Args:
            action (np.ndarray): Current action [Fx, Fy, Mz].
            previous_action (np.ndarray): Previous action [Fx, Fy, Mz].

        Returns:
            float: Total reward for the current step.
        """
        boat_pos = self.vessel.get_eta()[:2]
        boat_yaw = self.vessel.get_eta()[-1]

        # Goal position and size
        goal_x, goal_y, goal_size = self.goal

        # LOS Reward
        vector_to_goal = np.array([goal_y - boat_pos[0], goal_x - boat_pos[1]])
        chi = np.arctan2(vector_to_goal[1], vector_to_goal[0])  # LOS angle
        r_los = 10.0 * (np.pi / 2 - abs(chi - boat_yaw))  # λ = 10

        # Obstacle Reward
        r_obstacle = 0.0
        for obs_x, obs_y, obs_size in self.obstacles:
            distance_to_obstacle = np.linalg.norm(boat_pos - np.array([obs_y, obs_x]))
            if distance_to_obstacle < obs_size:
                r_obstacle += -100.0  # Strong penalty for collisions
            else:
                r_obstacle += -5.0 / max(distance_to_obstacle, 1e-5)  # σ = -5.0

        # Terminal Reward
        if (goal_x - goal_size / 2 <= boat_pos[1] <= goal_x + goal_size / 2 and
            goal_y - goal_size / 2 <= boat_pos[0] <= goal_y + goal_size / 2):
            r_terminal = 100.0  # Reached goal
        elif any(np.linalg.norm(boat_pos - np.array([obs_y, obs_x])) < obs_size for obs_x, obs_y, obs_size in self.obstacles):
            r_terminal = -100.0  # Collision
        else:
            r_terminal = 0.0

        # Action Penalty
        delta_action = np.linalg.norm(action - previous_action)
        r_action = -1.0 * delta_action  # α = -1.0

        # Total reward
        reward = r_los + r_obstacle + r_terminal + r_action
        return reward


