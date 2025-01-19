import sys
sys.path.append('./CSADtesting')  # Adjust if needed
import numpy as np
import time

from Environment.GridBoatEnv import GridWaveEnvironment
from Controller.adaptive_controller import MRACShipController

def main():
    """Run a simulation using the GridWaveEnvironment with a combined MRAC heading + surge PID controller."""

    # --- 1) Define how the goal should move over time (optional) ---
    def goal_func(t):
        # Example: a slowly moving goal in north/east
        north0, east0, size0 = 4, 12, 1
        new_north = north0 + 1.0 * np.sin(0.2 * t)
        new_east  = east0  + 0.5 * np.cos(0.1 * t)
        return (new_north, new_east, size0)

    # No moving obstacles in this example, so let's just keep them static or None
    obstacle_func = None

    # Time step
    dt = 0.1  

    # Create environment
    env = GridWaveEnvironment(
        dt=dt,
        grid_width=15,
        grid_height=6,
        render_on=True,
        final_plot=True
    )

    # Starting boat pose: (north=2, east=2, heading=90 deg) so it initially faces east
    start_pos = (2, 2, 90)

    # Initial wave conditions and obstacles
    wave_conditions = (1, 4.5, 0)  # (Hs=1, Tp=4.5, wave_dir=0 deg)
    initial_goal = (4, 12, 1)
    initial_obstacles = [(2, 7, 1.0)]

    env.set_task(
        start_position=start_pos,
        goal=initial_goal,
        wave_conditions=wave_conditions,
        obstacles=None,
        goal_func=goal_func,
        obstacle_func=None
    )

    # Create our MRAC-based controller
    controller = MRACShipController(dt=dt)

    simtime = 150.0
    max_steps = int(simtime / dt)

    print("Starting simulation...")
    start_time = time.time()

    for step_count in range(max_steps):
        # 1) Get current environment state
        state, done, info, reward = env.step([0, 0, 0])  
        # The above might just do a zero action as a placeholder to update the env one step.
        # Actually, let's re-check the doc: Usually you'd do `env.step(action)` once per loop,
        # so let's remove that first zero-step and do the real control step below.

        # Re-do: We'll just read the state *before* stepping:
        state = env.get_state()

        # 2) Compute control action
        #    Our environment's "goal" might be updated by goal_func(t).
        #    Let's fetch it from 'state["goal"]', if the environment sets that.
        current_goal = state["goal"]  # (n, e, size)
        goal_n, goal_e, _ = current_goal

        action = controller.compute_action(state, (goal_n, goal_e))

        # 3) Step the environment with that action
        new_state, done, info, reward = env.step(action)

        # 4) Check distance to goal
        boat_n, boat_e = new_state["boat_position"]
        distance_to_goal = np.sqrt((goal_n - boat_n)**2 + (goal_e - boat_e)**2)
        if distance_to_goal < 0.5:
            print(f"Goal reached at step {step_count} (distance={distance_to_goal:.2f}).")

        if done:
            # Environment signaled termination (collision/goal)
            break

    total_time = time.time() - start_time
    print(f"Wall-clock time: {total_time:.2f} s")
    print(f"Simulation speed: {(simtime / total_time):.2f}x real-time")
    print("Simulation completed.")

    env.plot_trajectory()

if __name__ == "__main__":
    main()
