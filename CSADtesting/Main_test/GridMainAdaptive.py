#------------------------------------------
# Step 1 Import Necessary Libraries
#------------------------------------------
import sys
import numpy as np
import time
# Adjust path if needed so Python finds your Environment folder, etc.
sys.path.append('./CSADtesting')
from CSADtesting.Environment.GridBoatEnv import GridWaveEnvironment
from CSADtesting.Controller.feedback_pd import feedback_linearizing_pd_controller ##This is not (at all) ideal for station keeping applications
from MCSimPython.utils import Rz, six2threeDOF, three2sixDOF
from MCSimPython.control.basic import PID
from CSADtesting.allocation.allocation import CSADThrusterAllocator
def main():
    # Simulation time step
    dt = 0.1  
    # Total simulation time, steps
    simtime = 300
    max_steps = int(simtime / dt)

    # Start pose 
    start_pos = (2, 2, 0)

    # Initial wave conditions 
    wave_conditions = (3, 20, 0)
    # Create environment
    env = GridWaveEnvironment(
        dt=dt,
        grid_width=15,
        grid_height=6,
        render_on=True,    # True => use pygame-based rendering
        final_plot=True    # True => at the end, produce a matplotlib plot of the trajectory
    )
    env.set_task(
        start_position=start_pos,
        wave_conditions=wave_conditions,
        four_corner_test=True,
        simtime=simtime
    )
    # Create the PD-based controller
    controller=PID(kp=[100.0, 100.0, 5.0], kd=[100.0, 100.0, 100.0], ki=[1.0, 1.0, 1.0])
    # Start the simulation
    print("Starting simulation...")
    start_time = time.time()

    for step_count in range(max_steps):
        eta_d, nu_d, eta_d_ddot, nu_d_body = env.get_four_corner_nd(step_count)
        state = env.get_state()
        nu_d = Rz(state["eta"][-1]) @ nu_d
        tau = controller.get_tau(eta=six2threeDOF(state["eta"]),eta_d=eta_d, nu= state["nu"], nu_d=nu_d)

        u = CSADThrusterAllocator().allocate(tau[0], tau[1], tau[2])
        #print(eta_d, tau)
        _, done, info, _ = env.step(action = u)
        if done:
            # The environment signaled termination (goal reached w/ heading or collision)
            print("Environment returned done; stopping simulation, because", info)
            break
    total_time = time.time() - start_time
    print(f"Wall-clock time: {total_time:.2f} s")
    print(f"Simulation speed: {(simtime / total_time):.2f}x real-time")
    print("Simulation completed.")
    # After finishing, if final_plot=True, plot the boat trajectory
    env.plot_trajectory()
        


if __name__ == "__main__":
    main()
