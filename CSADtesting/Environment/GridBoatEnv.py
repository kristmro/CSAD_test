import numpy as np
import matplotlib.pyplot as plt

# Only import pygame if needed
try:
    import pygame
except ImportError:
    pygame = None

from MCSimPython.simulator.csad import CSAD_DP_6DOF
from MCSimPython.waves.wave_loads import WaveLoad
from MCSimPython.waves.wave_spectra import JONSWAP
from MCSimPython.utils import three2sixDOF

class GridWaveEnvironment:
    """Grid-based Wave Environment with real-time rendering."""

    def __init__(
        self,
        dt=0.1,
        grid_width=15,
        grid_height=6,
        render_on=False,
        final_plot=True
    ):
        self.dt = dt
        self.grid_width = grid_width
        self.grid_height = grid_height

        self.vessel = CSAD_DP_6DOF(dt)
        self.waveload = None
        self.simulation_time = 0.0

        self.render_on = render_on
        self.final_plot = final_plot
        self.trajectory = []

        self.start_position = None
        self.goal = None
        self.obstacles = []
        self.wave_conditions = None

        self.goal_func = None
        self.obstacle_func = None

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

    def set_task(self, 
                 start_position, 
                 goal, 
                 wave_conditions, 
                 obstacles=None,
                 goal_func=None,
                 obstacle_func=None):
        self.start_position = start_position  # (north, east, heading_degs)
        self.goal = goal                      # (north, east, size)
        self.wave_conditions = wave_conditions
        self.obstacles = obstacles if obstacles else []
        
        self.goal_func = goal_func
        self.obstacle_func = obstacle_func

        self.reset()

    def reset(self):
        self.vessel = CSAD_DP_6DOF(self.dt, method="RK4")
        self.simulation_time = 0.0

        north0, east0, heading_deg = self.start_position
        eta_start = three2sixDOF(
            np.array([north0, east0, np.deg2rad(heading_deg)])
        )
        self.vessel.set_eta(eta_start)

        self.set_wave_conditions(*self.wave_conditions)

        if self.final_plot:
            self.trajectory = []

        self.previous_action = np.zeros(3)

        if self.render_on and self.screen is not None:
            self.screen.fill((20, 20, 20))

    def set_wave_conditions(self, hs, tp, wave_dir_deg):
        self.wave_conditions = (hs, tp, wave_dir_deg)
        wp = 2 * np.pi / tp
        N_w = 100
        wmin, wmax = wp / 2, 3. * wp
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

    def step(self, action):
        self.simulation_time += self.dt

        if self.goal_func is not None:
            self.goal = self.goal_func(self.simulation_time)
        if self.obstacle_func is not None:
            self.obstacles = self.obstacle_func(self.simulation_time)

        tau_6dof = np.zeros(6)
        tau_6dof[:2] = action[:2]
        tau_6dof[-1] = action[-1]

        tau_wave = self.waveload(self.simulation_time, self.vessel.get_eta())
        self.vessel.integrate(0, 0, tau_6dof + tau_wave)

        boat_pos = self.vessel.get_eta()[:2]
        done, info = self._check_termination(boat_pos)

        reward = self.compute_reward(action, self.previous_action)
        self.previous_action = action

        if self.final_plot:
            self.trajectory.append(boat_pos.copy())

        if self.render_on:
            self.render()

        return self.get_state(), done, info, reward

    def _check_termination(self, boat_pos):
        # Compute boat hull in global coords
        hull_local = self._get_boat_hull_local_pts()
        boat_yaw = self.vessel.get_eta()[5]
        c, s = np.cos(boat_yaw), np.sin(boat_yaw)
        rot = np.array([[c, s], [-s, c]])
        hull_global = []
        for (lx, ly) in hull_local:
            gx, gy = rot @ np.array([lx, ly])
            hull_global.append(np.array([boat_pos[0] + gy, boat_pos[1] + gx]))

        # Check goal as bounding box
        g_n, g_e, g_size = self.goal
        for pt in hull_global:
            if (g_n - g_size/2 <= pt[0] <= g_n + g_size/2 and
                g_e - g_size/2 <= pt[1] <= g_e + g_size/2):
                print("Goal reached!")
                return True, {"reason": "goal_reached"}

        # Check obstacles as circle collisions
        for obs_n, obs_e, obs_size in self.obstacles:
            for pt in hull_global:
                dist = np.linalg.norm(pt - np.array([obs_n, obs_e]))
                if dist < obs_size / 2.0:
                    print("Collision with obstacle!")
                    return True, {"reason": "collision"}

        return False, {}

        return False, {}

    def get_state(self):
        eta = self.vessel.get_eta()  # 6DOF
        return {
            "boat_position": eta[:2],
            "boat_orientation": eta[-1],
            "goal": self.goal,
            "obstacles": self.obstacles,
            "wave_conditions": self.wave_conditions
        }

    def render(self):
        if not self.render_on or self.screen is None:
            return

        self.screen.fill((20, 20, 20))
        self._draw_grid()
        self._draw_goal()
        self._draw_obstacles()
        self._draw_boat()

        pygame.display.flip()
        self.clock.tick(1000)

    def _draw_grid(self):
        if not self.render_on or self.screen is None:
            return
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
        if not self.render_on or self.screen is None:
            return
        import pygame

        g_n, g_e, g_size = self.goal
        px = g_e * self.x_scale
        py = (self.grid_height - g_n) * self.y_scale
        radius_px = (g_size / 2) * self.x_scale

        pygame.draw.circle(
            self.screen,
            (255, 215, 0),
            (int(px), int(py)),
            int(radius_px)
        )

    def _draw_obstacles(self):
        if not self.render_on or self.screen is None:
            return
        import pygame

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
        - y is 'forward' (bow direction)
        - x is 'starboard'
        This matches the original snippet’s (x=forward, y=starboard),
        but we then swap so that snippet_x -> local_y, snippet_y -> local_x.
        """
        Lpp = 2.5780001
        B   = 0.4440001
        halfL = 0.5 * Lpp
        halfB = 0.5 * B

        bow_start_x = 0.9344  # along the forward axis in the snippet

        def bow_curve_points(n=40):
            pts = []
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

        # Build the hull in snippet coords: (x=forward, y=starboard)
        hull_pts_snippet = []
        # Top edge
        hull_pts_snippet.append((x_stern_left, +halfB))
        hull_pts_snippet.append((x_stern_right, +halfB))
        # Bow curve
        hull_pts_snippet.extend(bow_curve_points(n=40))
        # Bottom edge
        hull_pts_snippet.append((x_stern_left, -halfB))
        # Close
        hull_pts_snippet.append((x_stern_left, +halfB))

        # Convert snippet → local: local_x = snippet_y, local_y = snippet_x
        hull_pts_local = [
            (pt_snip[1], pt_snip[0])  # (x_local, y_local) = (y_snip, x_snip)
            for pt_snip in hull_pts_snippet
        ]

        return np.array(hull_pts_local)


    def _draw_boat(self):
        """
        Draw the boat so that heading=0 means boat faces *up* (north),
        90 means right (east), 180=down (south), 270=left (west).
        """
        if not self.render_on or self.screen is None:
            return
        import pygame

        # Current boat state
        eta = self.vessel.get_eta()
        boat_n, boat_e, boat_yaw = eta[0], eta[1], eta[5]  # [north, east, yaw_radians]

        # 1) Get local hull shape
        hull_local = self._get_boat_hull_local_pts()

        # 2) Rotation matrix for heading 'boat_yaw':
        #    Because heading=0 => north,  heading=90 => east, etc.
        #    R(h) = [[ cos(h), sin(h)],
        #            [-sin(h), cos(h)]]
        c = np.cos(boat_yaw)
        s = np.sin(boat_yaw)
        rot = np.array([
            [ c,  s],
            [-s,  c]
        ])

        # 3) Transform local points → global → screen
        pixel_pts = []
        for (lx, ly) in hull_local:
            # Rotate
            gx, gy = rot @ np.array([lx, ly])
            # Shift by boat's global position
            gx += boat_e  # global east
            gy += boat_n  # global north
            # Convert to screen
            sx = int(gx * self.x_scale)
            sy = int((self.grid_height - gy) * self.y_scale)
            pixel_pts.append((sx, sy))

        # 4) Draw hull polygon
        pygame.draw.polygon(self.screen, (0, 100, 255), pixel_pts)

    def compute_reward(self, action, prev_action):
        return 0.0

    def close(self):
        if self.render_on and self.screen is not None and pygame is not None:
            pygame.quit()

    def __del__(self):
        self.close()
        
    def plot_trajectory(self):
        if not self.final_plot:
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
        plt.title("Boat Trajectory (15×6 Domain)")
        plt.legend(loc='upper right', fontsize='small', scatterpoints=1, markerscale=0.1)
        plt.grid(True)
        plt.show()
