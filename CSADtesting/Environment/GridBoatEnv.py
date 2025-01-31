import numpy as np
import matplotlib.pyplot as plt

import pygame

from MCSimPython.simulator.csad import CSAD_DP_6DOF
from MCSimPython.waves.wave_loads import WaveLoad
from MCSimPython.waves.wave_spectra import JONSWAP
from CSADtesting.filters.reference_filter import ThrdOrderRefFilter
from MCSimPython.utils import three2sixDOF, six2threeDOF, Rz

class GridWaveEnvironment:
    """Grid-based Wave Environment with real-time rendering and a simple goal/obstacle framework.

    Parameters
    ----------
    dt : float
        Simulation time-step.
    grid_width : float
        Width of the domain (east direction).
    grid_height : float
        Height of the domain (north direction).
    render_on : bool
        If True, use pygame to render in real-time.
    final_plot : bool
        If True, store the trajectory and produce a matplotlib plot at the end.
    """

    def __init__(
        self,
        dt=0.1,
        grid_width=15,
        grid_height=6,
        render_on=False,
        final_plot=True,
        vessel=CSAD_DP_6DOF,
    ):
        self.dt = dt
        self.grid_width = grid_width
        self.grid_height = grid_height

        # Create the vessel simulator
        self.vessel = vessel(dt)
        self.waveload = None
        self.curr_sim_time = 0.0

        # Rendering / plotting
        self.render_on = render_on
        self.final_plot = final_plot
        self.trajectory = []

        # Task-related
        self.start_position = None
        self.goal = None
        self.obstacles = []
        self.wave_conditions = None
        self.goal_func = None
        self.obstacle_func = None

        # FOUR-CORNER TEST
        self.four_corner_test = False
        self.set_points = None
        self.simtime = 300
        self.t = None
        self.ref_model = None
        self.store_xd = None

        # Tolerances and optional goal heading
        self.position_tolerance = 0.5  # meters
        self.goal_heading_deg = None   # if None, no heading requirement
        self.heading_tolerance_deg = 10.0  # degrees

        # Pygame setup if rendering
        self.screen = None
        self.clock = None
        self.WINDOW_WIDTH = 750
        self.WINDOW_HEIGHT = 300
        if self.render_on and pygame is not None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
            pygame.display.set_caption("Boat Navigation Simulation")
            self.clock = pygame.time.Clock()

        self.x_scale = (self.WINDOW_WIDTH  / self.grid_width)  
        self.y_scale = (self.WINDOW_HEIGHT / self.grid_height) 
    def get_vessel(self):
        return self.vessel
    
    def set_task(self, 
                 start_position, 
                 goal=None, 
                 wave_conditions=None, 
                 obstacles=None,
                 goal_func=None,
                 obstacle_func=None,
                 position_tolerance=0.5,
                 goal_heading_deg=None,
                 heading_tolerance_deg=10.0,
                 four_corner_test=False,
                 simtime=300):
        """
        Configure the environment to a new task.

        Parameters
        ----------
        start_position : tuple (north, east, heading_deg)
            Initial position and heading of the boat in the domain.
        goal : tuple (north, east, size)
            The goal region (center north/east, plus a "size" for drawing).
        wave_conditions : tuple (Hs, Tp, wave_dir_deg)
            Significant wave height, peak period, and wave direction (deg).
        obstacles : list of tuples [(obs_n, obs_e, obs_size), ...] or None
            Each obstacle is a circle in the domain with center (obs_n, obs_e) 
            and diameter 'obs_size'.
        goal_func : callable or None
            If not None, a function f(t) -> (g_n, g_e, g_size) defining how the goal
            changes over time.
        obstacle_func : callable or None
            If not None, a function f(t) -> list_of_obstacles, for dynamic obstacles.
        position_tolerance : float
            The maximum distance from the boat center to the goal center at which
            we consider the goal 'reached'.
        goal_heading_deg : float or None
            If not None, also require the boat heading to be within heading_tolerance_deg
            of this heading to consider the goal reached.
        heading_tolerance_deg : float
            Allowable heading deviation in degrees from goal_heading_deg.

        FOR FOUR-CORNER TEST:
        --------------------------------
        TODO
        """
        self.start_position = start_position
        self.goal = goal
        self.wave_conditions = wave_conditions
        self.obstacles = obstacles if obstacles else []
        
        self.goal_func = goal_func
        self.obstacle_func = obstacle_func
        
        # Store the tolerances and optional heading requirement
        self.position_tolerance = position_tolerance
        self.goal_heading_deg = goal_heading_deg
        self.heading_tolerance_deg = heading_tolerance_deg

        self.four_corner_test = four_corner_test
        if self.four_corner_test:
            self.set_points = [np.array([2.0, 2.0, 0.0]),
                  np.array([4.0, 2.0, 0.0]),
                  np.array([4.0, 4.0, 0.0]),
                  np.array([4.0, 4.0, -np.pi/4]),
                  np.array([2.0, 4.0, -np.pi/4]),
                  np.array([2.0, 2.0, 0.0])
            ]
            self.simtime = simtime
            self.t= np.arange(0, self.simtime, self.dt)
            print("Four-corner test enabled, and simulation time NEED TO BE SET TO 300s")
            self.ref_model = ThrdOrderRefFilter(self.dt, initial_eta=self.start_position)
            self.store_xd = np.zeros((len(self.t), 9)) # Store the desired trajectory 3columns for pos and 3 for vel and 3 for acc
        

        # Reset environment to this new setup
        self.reset()

    def reset(self):
        """Resets the environment and vessel state."""
        self.vessel = CSAD_DP_6DOF(self.dt, method="RK4")
        self.curr_sim_time = 0.0

        north0, east0, heading_deg = self.start_position
        eta_start = three2sixDOF(
            np.array([north0, east0, np.deg2rad(heading_deg)])
        )
        self.vessel.set_eta(eta_start)

        self.set_wave_conditions(*self.wave_conditions)

        if self.final_plot:
            self.trajectory = []

        # For reward shaping
        self.previous_action = np.zeros(3)

        if self.render_on and self.screen is not None:
            self.screen.fill((20, 20, 20))

    def set_wave_conditions(self, hs, tp, wave_dir_deg, N_w=100, gamma=3.3):
        """Builds a wave load from the specified wave parameters."""
        self.wave_conditions = (hs, tp, wave_dir_deg)
        wp = 2 * np.pi / tp
        wmin, wmax = wp / 2, 3. * wp
        dw = (wmax - wmin) / N_w
        w = np.linspace(wmin, wmax, N_w, endpoint=True)

        jonswap = JONSWAP(w)
        freq, spec = jonswap(hs, tp, gamma)

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
    def get_four_corner_nd(self, step_count):
        """Four-corner test for the environment."""
        print(self.t[step_count])
        if self.t[step_count] > 250:
            self.ref_model.set_eta_r(self.set_points[5])
        elif self.t[step_count] > 200:
            self.ref_model.set_eta_r(self.set_points[4])
        elif self.t[step_count] > 150:
            self.ref_model.set_eta_r(self.set_points[3])
        elif self.t[step_count] > 100:
            self.ref_model.set_eta_r(self.set_points[2])
        elif self.t[step_count] > 50:
            self.ref_model.set_eta_r(self.set_points[1])
        else:
            self.ref_model.set_eta_r(self.set_points[0])
        # Update reference model
        self.ref_model.update()
        
        # Extract reference states
        eta_d = self.ref_model.get_eta_d()      # Desired position & heading (global frame)
        eta_d_dot = self.ref_model.get_eta_d_dot()  # Desired velocity (global frame)
        eta_d_ddot = self.ref_model.get_eta_d_ddot()  # Desired acceleration (global frame)
        nu_d_body = self.ref_model.get_nu_d()        # Desired velocity in body frame
    
        # Store for debugging or plotting
        self.store_xd[step_count] = self.ref_model._x

        return eta_d, eta_d_dot, eta_d_ddot, nu_d_body

    def step(self, action):
        """Performs one simulation step given the control action."""
        self.curr_sim_time += self.dt

        # Possibly update goal/obstacles over time
        if self.goal_func is not None:
            self.goal = self.goal_func(self.curr_sim_time)
        if self.obstacle_func is not None:
            self.obstacles = self.obstacle_func(self.curr_sim_time)
        

        # Convert 3DOF action -> 6DOF (tau surge, sway, yaw)
        tau_6dof = three2sixDOF(action)

        # Wave forces
        tau_wave = self.waveload(self.curr_sim_time, self.vessel.get_eta())

        # Integrate vessel with control + wave
        self.vessel.integrate(0, 0, tau_6dof + tau_wave)

        boat_pos = self.vessel.get_eta()[:2]
        done, info = self._check_termination(boat_pos)

        # Simple reward function (placeholder)
        reward = self.compute_reward(action, self.previous_action)
        self.previous_action = action

        if self.final_plot:
            self.trajectory.append(boat_pos.copy())

        if self.render_on:
            self.render()

        return self.get_state(), done, info, reward

    def _check_termination(self, boat_pos):
        """
        if self.four_corner_test:
            then just check if the simulation time is over
        
        if self.goal is not None:
            Returns (done, info).

            'done' is True if:
            1) The boat is within self.position_tolerance of the goal center, AND
                if self.goal_heading_deg is set, the boat heading is within
                self.heading_tolerance_deg of that heading.
            2) The boat collides with an obstacle (circle check against hull points).
            """
        if self.four_corner_test:
            if self.curr_sim_time > self.simtime:
                print("Four-corner test completed.")
                return True, {"reason": "four_corner_test_completed"}
            else:
                return False, {}
        if self.goal is not None:
            boat_yaw = self.vessel.get_eta()[5]  # in radians
            g_n, g_e, g_size = self.goal

            # 1) Check distance-based goal attainment
            dx = boat_pos[0] - g_n
            dy = boat_pos[1] - g_e
            distance_to_goal = np.sqrt(dx**2 + dy**2)

            heading_ok = True
            if self.goal_heading_deg is not None:
                boat_heading_deg = np.rad2deg(boat_yaw)
                heading_diff = boat_heading_deg - self.goal_heading_deg
                # Normalize difference to [-180, 180]
                heading_diff = (heading_diff + 180) % 360 - 180
                heading_ok = abs(heading_diff) <= self.heading_tolerance_deg

            if distance_to_goal < self.position_tolerance and heading_ok:
                print("Goal reached!")
                return True, {"reason": "goal_reached"}

            # 2) Check obstacle collisions (circle collision test on hull points)
            hull_local = self._get_boat_hull_local_pts()
            c, s = np.cos(boat_yaw), np.sin(boat_yaw)
            rot = np.array([[c, s], [-s, c]])

            # Convert local hull to global points
            hull_global = []
            for (lx, ly) in hull_local:
                gx, gy = rot @ np.array([lx, ly])
                gx_global = boat_pos[0] + gy  # recall boat_pos = [n, e]
                gy_global = boat_pos[1] + gx
                hull_global.append(np.array([gx_global, gy_global]))

            for obs_n, obs_e, obs_size in self.obstacles:
                obs_radius = obs_size / 2.0
                for pt in hull_global:
                    dist = np.linalg.norm(pt - np.array([obs_n, obs_e]))
                    if dist < obs_radius:
                        print("Collision with obstacle!")
                        return True, {"reason": "collision"}

            return False, {}
        else:
            
            return True, {"reason": "No goal or four_corner_test initiated breaking the environment"}

    def get_state(self):
        """Return a dictionary describing the current environment state."""
        eta = self.vessel.get_eta()  # 6DOF
        nu = six2threeDOF(self.vessel.get_nu()) # Vel 6DOF->3DOF
        return {
            "eta": eta,                 # (n, e, d, u, v, r) in 6DOF
            "nu": nu,             # (u, v, r) in 3DOF
            "goal": self.goal,
            "obstacles": self.obstacles,
            "wave_conditions": self.wave_conditions,


        }

    def render(self):
        """Render the environment using pygame (if available)."""
        if not self.render_on or self.screen is None:
            return

        self.screen.fill((20, 20, 20))
        self._draw_grid()
        if self.goal is not None or self.goal_func is not None:
            self._draw_goal()
        if self.obstacles is not None or self.obstacle_func is not None:
            self._draw_obstacles()
        self._draw_boat()

        pygame.display.flip()
        self.clock.tick(1000)

    def _draw_grid(self):
        """Draw a basic grid on the pygame surface."""
        if not self.render_on or self.screen is None:
            return

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
        """Draw the goal region as a circle."""
        if not self.render_on or self.screen is None:
            return

        g_n, g_e, g_size = self.goal
        px = g_e * self.x_scale
        py = (self.grid_height - g_n) * self.y_scale
        radius_px = (g_size / 2) * self.x_scale

        pygame.draw.circle(
            self.screen,
            (255, 215, 0),  # gold color
            (int(px), int(py)),
            int(radius_px)
        )

    def _draw_obstacles(self):
        """Draw all obstacles as red circles."""
        if not self.render_on or self.screen is None:
            return

        for obs_n, obs_e, obs_size in self.obstacles:
            px = obs_e * self.x_scale
            py = (self.grid_height - obs_n) * self.y_scale
            radius_px = (obs_size / 2) * self.x_scale

            pygame.draw.circle(
                self.screen,
                (200, 0, 0),
                (int(px), int(py)),
                int(radius_px)
            )

    def _get_boat_hull_local_pts(self):
        """
        Return hull points (rect-stern + single Bézier bow) in local (x,y):
         - local x = 'starboard'
         - local y = 'forward'
        so that heading=0 means the boat is pointing 'north' in the global frame.
        """
        Lpp = 2.5780001
        B   = 0.4440001
        halfL = 0.5 * Lpp
        halfB = 0.5 * B

        bow_start_x = 0.9344  # forward from stern

        def bow_curve_points(n=40):
            """Bezier curve from the wide part of the bow to the bow tip."""
            pts = []
            # Points in snippet's coordinate (x=forward, y=starboard)
            P0 = (bow_start_x, +halfB)
            P1 = (halfL, 0.0)
            P2 = (bow_start_x, -halfB)
            for i in range(n+1):
                s = i / n
                x = (1 - s)**2 * P0[0] + 2*(1 - s)*s * P1[0] + s**2 * P2[0]
                y = (1 - s)**2 * P0[1] + 2*(1 - s)*s * P1[1] + s**2 * P2[1]
                pts.append((x, y))
            return pts

        x_stern_left  = -halfL
        x_stern_right = bow_start_x

        hull_pts_snippet = []
        # top edge
        hull_pts_snippet.append((x_stern_left, +halfB))
        hull_pts_snippet.append((x_stern_right, +halfB))
        # bow curve
        hull_pts_snippet.extend(bow_curve_points(n=40))
        # bottom edge
        hull_pts_snippet.append((x_stern_left, -halfB))
        # close
        hull_pts_snippet.append((x_stern_left, +halfB))

        # Convert snippet → local (x_local= snippet_y, y_local= snippet_x)
        hull_pts_local = [
            (pt_snip[1], pt_snip[0])
            for pt_snip in hull_pts_snippet
        ]

        return np.array(hull_pts_local)

    def _draw_boat(self):
        """Draw the boat polygon on the pygame surface, heading=0 => boat faces north."""
        if not self.render_on or self.screen is None:
            return

        eta = self.vessel.get_eta()
        boat_n, boat_e, boat_yaw = eta[0], eta[1], eta[-1]

        hull_local = self._get_boat_hull_local_pts()

        c = np.cos(boat_yaw)
        s = np.sin(boat_yaw)
        rot = np.array([
            [ c,  s],
            [-s,  c]
        ])

        pixel_pts = []
        for (lx, ly) in hull_local:
            gx, gy = rot @ np.array([lx, ly])
            gx += boat_e
            gy += boat_n
            sx = int(gx * self.x_scale)
            sy = int((self.grid_height - gy) * self.y_scale)
            pixel_pts.append((sx, sy))

        # Color the hull polygon
        pygame.draw.polygon(self.screen, (0, 100, 255), pixel_pts)

    def compute_reward(self, action, prev_action):
        """Compute a reward for this step (placeholder)."""
        # You can expand this logic as needed.
        return 0.0

    def close(self):
        """Close environment."""
        if self.render_on and self.screen is not None and pygame is not None:
            pygame.quit()

    def __del__(self):
        self.close()
        
    def plot_trajectory(self):
        """If final_plot=True, visualize the path taken by the boat."""
        if not self.final_plot:
            return
        
        if self.goal_func is not None or self.goal is not None:
            traj = np.array(self.trajectory)
            plt.figure(figsize=(8, 4))
            plt.plot(traj[:, 1], traj[:, 0], 'g-', label="Boat Trajectory")

            g_n, g_e, g_s = self.goal
            plt.scatter(g_e, g_n, c='yellow',
                        s=(g_s * self.x_scale)**2,
                        edgecolor='black', label="Goal")

            for obs_n, obs_e, obs_size in self.obstacles:
                plt.scatter(obs_e, obs_n, c='red',
                            s=(obs_size * self.x_scale)**2,
                            edgecolor='black', label="Obstacle")

            plt.xlim([0, self.grid_width])
            plt.ylim([0, self.grid_height])
            plt.xlabel("East [m]")
            plt.ylabel("North [m]")
            plt.title("Boat Trajectory ({}×{} Domain)".format(self.grid_width, self.grid_height))
            plt.legend(loc='upper right', fontsize='small', scatterpoints=1, markerscale=0.1)
            plt.grid(True)
            plt.show()
        if self.four_corner_test:
            traj = np.array(self.trajectory)
            plt.figure(figsize=(8, 4))
            plt.plot(traj[:, 1], traj[:, 0], 'b-', label="Boat Trajectory")
            plt.plot(self.store_xd[:, 1], self.store_xd[:, 0], 'g-', label="Desired Trajectory")
            plt.xlim([0, self.grid_width])
            plt.ylim([0, self.grid_height])
            plt.xlabel("East [m]")
            plt.ylabel("North [m]")
            plt.title("Desired Trajectory and real trajectory ({}×{} Domain)".format(self.grid_width, self.grid_height))
            plt.legend(loc='upper right', fontsize='small', scatterpoints=1, markerscale=0.1)
            plt.grid(True)
            plt.show()

            # Plot the desired trajectory over time
            plt.figure(figsize=(8, 4))
            plt.plot(self.t, self.store_xd[:, 0], 'r-', label="North")
            plt.xlabel("Time [s]")
            plt.ylabel("Position [m]")
            plt.title("Desired trajectory over time")  
            plt.legend(loc='upper right', fontsize='small', scatterpoints=1, markerscale=0.1)


