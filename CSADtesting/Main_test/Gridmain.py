import sys
sys.path.append('./CSADtesting')  # Adjust this path as needed
import numpy as np
import time
from Environment.GridBoatEnv import GridWaveEnvironment  # Make sure your file name/path is correct

def main():
    """Run a simulation using the GridWaveEnvironment with user-defined time-dependent goal/obstacles."""

    # --- 1) Define how the goal should move over time ---
    #    goal_func(t) should return (north, east, size) at simulation time 't'
    def goal_func(t):
        # For example, place the goal near the top-right corner and make it move slightly
        north0, east0, size0 = 4, 12, 1
        new_north = north0 + 1.0 * np.sin(0.2 * t)  # sinusoidal motion in north
        new_east  = east0  + 0.5 * np.cos(0.1 * t)  # sinusoidal motion in east
        return (new_north, new_east, size0)

    # --- 2) Define how the obstacles move over time ---
    #    obstacle_func(t) should return a list of (north, east, size) for each obstacle
    def obstacle_func(t):
        # Example: one obstacle or multiple obstacles
        # Let’s make one obstacle swirl around (2,7) with small amplitude
        obs1_north0, obs1_east0, obs_size = 2, 7, 1
        obs1_north = obs1_north0 + 0.5 * np.sin(0.3 * t)
        obs1_east  = obs1_east0  + 0.5 * np.cos(0.3 * t)

        # If desired, define more obstacles similarly
        #   e.g., second obstacle
        # obs2_north = ...
        # obs2_east  = ...
        # obs2_size  = ...
        # return [(obs1_north, obs1_east, obs_size),
        #         (obs2_north, obs2_east, obs2_size)]

        return [(obs1_north, obs1_east, obs_size)]

    # Time step
    dt = 0.1  

    # Create environment with a 15 (width) x 6 (height) grid
    env = GridWaveEnvironment(
        dt=dt,
        grid_width=15,
        grid_height=6,
        render_on=True,
        final_plot=True
    )

    # Starting boat pose: (north=0, east=2, heading=0 deg),
    # where heading=0 means the boat points *upward* on screen (because of our offset).
    start_pos = (2, 2, 90)

    # We still pass an *initial* goal and obstacles, but they’ll be overridden each step by the user-defined functions.
    initial_goal = (4, 12, 0.5)
    wave_conditions = (1, 4.5, 0)  # (Hs, Tp, wave_dir_deg)
    initial_obstacles = [(2, 7, 0.5)]

    # Set the task in the environment, providing the user-defined update functions
    env.set_task(
        start_position=start_pos,
        goal=initial_goal,
        wave_conditions=wave_conditions,
        obstacles=initial_obstacles,
        goal_func=goal_func,
        obstacle_func=obstacle_func
    )

    # Simulation length
    simtime = 80.0
    max_steps = int(simtime / dt)

    # A constant action (Fx, Fy, Mz)
    pre_action = np.array([0, 0, 0.0])

    print("Starting simulation...")
    start_time = time.time()

    for step_count in range(max_steps):
        state, done, info, reward = env.step(pre_action)

        
        if done:
            break

    total_time = time.time() - start_time
    print(f"Wall-clock time: {total_time:.2f} seconds")
    print(f"Simulation speed: {simtime / total_time:.2f} × real-time")
    print("Simulation completed.")

    # Plot the trajectory
    env.plot_trajectory()

if __name__ == "__main__":
    main()
