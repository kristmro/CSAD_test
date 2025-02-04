# # ------------------------------------------------------------------------
# # Step 1 Import Necessary Libraries
# # ------------------------------------------------------------------------

# import pickle
# import numpy as np
# from tqdm import tqdm
# from scipy.signal import lti, lsim
# from MCSimPython.simulator.csad import CSAD_DP_6DOF
# from MCSimPython.waves.wave_loads import WaveLoad
# from MCSimPython.waves.wave_spectra import JONSWAP
# from MCSimPython.utils import three2sixDOF, six2threeDOF
# #from CSADtesting.allocation.allocation import CSADThrusterAllocator

# # ------------------------------------------------------------------------
# # Step 2: Generate Smooth trajectories with smooth filter
# # ------------------------------------------------------------------------
# from MCSimPython.guidance.filter import ThrdOrderRefFilter
# import numpy as np
# import matplotlib.pyplot as plt

# plt.rcParams.update({
#     'figure.figsize': (8, 6),
#     'font.size': 12,
#     'font.family': 'serif',
#     'axes.grid': True
# })


# # Simulation settings
# sim_time = 300
# dt = 0.01
# t = np.arange(0, sim_time, dt)

# # Velocity coefficients
# omega = np.array([0.3, 0.3, 0.3])

# ref_model = ThrdOrderRefFilter(dt, omega, initial_eta=[2.0, 2.0, 0.0])

# # Set points
# set_points = [
#     np.array([2.0, 2.0, 0.0]),
#     np.array([2.0, 4.0, 0.0]),
#     np.array([4.0, 4.0, 0.0]),
#     np.array([4.0, 4.0, -np.pi/4]),
#     np.array([4.0, 2.0, -np.pi/4]),
#     np.array([2.0, 2.0, 0.0])
# ]



# x = np.zeros((len(t), 9))
# for i in range(1, len(t)):
#     if t[i] > 250:
#         ref_model.set_eta_r(set_points[5])
#     elif t[i] > 200:
#         ref_model.set_eta_r(set_points[4])
#     elif t[i] > 150:
#         ref_model.set_eta_r(set_points[3])
#     elif t[i] > 100:
#         ref_model.set_eta_r(set_points[2])
#     elif t[i] > 50:
#         ref_model.set_eta_r(set_points[1])
#     else:
#         ref_model.set_eta_r(set_points[0])
#     ref_model.update()
#     x[i] = ref_model._x


# ------------------------------------------------------------------------
# Step 3: PD Controller
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Step 4: Build wave environment
# ------------------------------------------------------------------------
# Define wave parameters




#------------------------------------------
# Step 1 Import Necessary Libraries
#------------------------------------------
import sys
import numpy as np
import time
# Adjust path if needed so Python finds your Environment folder, etc.
sys.path.append('./CSADtesting')
from CSADtesting.Environment.GridBoatEnv import GridWaveEnvironment
from CSADtesting.Controller.LF_ import feedback_linearizing_pd_controller ##This is not (at all) ideal for station keeping applications
from MCSimPython.utils import Rz, six2threeDOF, three2sixDOF
def main():
    # Simulation time step
    dt = 0.01  
    # Total simulation time, steps
    simtime = 120
    max_steps = int(simtime / dt)

    # Start pose 
    start_pos = (2, 2, 0)

    # Initial wave conditions 
    wave_conditions = (0.1, 20, 0)
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
    controller=feedback_linearizing_pd_controller
    # Start the simulation
    print("Starting simulation...")
    start_time = time.time()

    for step_count in range(max_steps):
        eta_d, nu_d, eta_d_ddot, nu_d_body = env.get_four_corner_nd(step_count)
        state = env.get_state()
        tau, u = controller(env.get_vessel(), six2threeDOF(state["eta"]), state["nu"], eta_d, nu_d_body, eta_d_ddot)
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
