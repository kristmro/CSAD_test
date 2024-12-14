
import sys
sys.path.append('./fast_adap_embedding')  # Adjust this path as needed
import numpy as np

from Envionment.GridBoatEnv import GridWaveEnvironment

def main():
    """Run a simulation using the GridWaveEnvironment."""
    # Initialize environment with rendering and final plotting options
    dt = 0.1  # Set your desired time step here
    env = GridWaveEnvironment(dt=dt, grid_size=100, render_on=True, final_plot=True)

    # Define a simple task
    start_pos = (40, 20, 0)  # Starting at (40, 20) facing north
    goal = (80, 80, 10)  # Goal centered at (80, 80) with a size of 10x10
    waves = (2.5, 12.5, 45)  # Wave conditions (Hs, Tp, direction in degrees)
    obstacles = [(50, 50, 10)]  # One obstacle at (50, 50) with size 10

    # Configure the task
    env.set_task(start_position=start_pos, goal=goal, wave_conditions=waves, obstacles=obstacles)

    # Initialize simulation variables
    #state = env.get_state()
    # Simulation parameters
    simtime = 80
    max_steps = int(simtime / dt)

    constant_action = np.array([0.0, 0.0, 0.0])  # Fx, Fy, Mz (body frame)

    print("Starting simulation...")
    for step_count in range(max_steps):
        #print(f"Step {step_count}")
        state, done, info, reward = env.step(constant_action)
        if done:
            break  # Exit the loop if the episode is over
    
    print("Simulation completed.")

    # Plot final trajectory if enabled
    env.plot_trajectory()

if __name__ == "__main__":
    main()
