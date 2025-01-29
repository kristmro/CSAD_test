

import numpy as np
import pickle
import sys

# -------------------------------------------------------------------------
# 1) Import your vessel & wave from MCSimPython
# -------------------------------------------------------------------------
from MCSimPython.simulator.csad import CSAD_DP_6DOF
from MCSimPython.waves.wave_spectra import JONSWAP
from MCSimPython.waves.wave_loads import WaveLoad
from MCSimPython.utils import three2sixDOF

# Import the new thruster-allocation class
from CSADtesting.allocation.allocation import CSADThrusterAllocator

# -------------------------------------------------------------------------
# 2) Feedback-Linearizing PD in 3 DOF
# -------------------------------------------------------------------------
def feedback_linearizing_pd_3dof(vessel, eta, nu, r, dr, ddr, kp=10.0, kd=0.1):
    """
    3-DOF PD in [n,e,psi]:
      e = q - r,
      de= dq - dr,
      dv= ddr - kp*e - kd*de,
      tau = M_3 dv + ... (ignoring advanced Coriolis/G)
    """
    idx = [0,1,5]  # surge, sway, yaw

    e  = np.array([eta[0]-r[0], eta[1]-r[1], eta[5]-r[2]])
    de = np.array([nu[0]-dr[0], nu[1]-dr[1], nu[5]-dr[2]])
    dv = ddr - kp*e - kd*de

    M_3 = vessel._M[np.ix_(idx, idx)]
    C_3 = np.zeros((3,3))  # simplified
    G_6 = np.zeros(6)      # simplified
    G_3 = G_6[idx]

    tau_3 = M_3 @ dv + C_3 @ nu[idx] + G_3
    return tau_3

# -------------------------------------------------------------------------
# 3) Build random wave
# -------------------------------------------------------------------------
def build_random_wave():
    """
    Example: random Hs, Tp, wave_dir => wave_load.
    """
    hs  = np.random.uniform(0.5, 2.0)
    tp  = np.random.uniform(4.0, 10.0)
    wdir_deg = np.random.uniform(0.0, 360.0)

    N_w = 80
    wmin = 2*np.pi/tp/2
    wmax = 3*(2*np.pi/tp)
    dw   = (wmax - wmin)/N_w
    w_arr= np.linspace(wmin, wmax, N_w, endpoint=True)

    jonswap = JONSWAP(w_arr)
    freq, spec = jonswap(hs, tp, gamma=3.3)

    wave_amps= np.sqrt(2*spec*dw)
    eps      = np.random.uniform(0,2*np.pi,N_w)
    wave_dir = np.ones(N_w)*np.deg2rad(wdir_deg)

    wave_load= WaveLoad(
        wave_amps=wave_amps,
        freqs=w_arr,
        eps=eps,
        angles=wave_dir,
        config_file='vessel_json.json',
        interpolate=True,
        qtf_method="geo-mean",
        deep_water=True
    )

    return wave_load, (hs, tp, wdir_deg)

# -------------------------------------------------------------------------
# 4) Main data-generation
# -------------------------------------------------------------------------
def main_generate_data():
    """
    Generate data with a 3-DOF PD,
    thruster allocation (CSADThrusterAllocator),
    wave loads, and a 4-corner path.
    """
    np.random.seed(0)

    dt      = 0.05
    T       = 40.0
    n_steps = int(T/dt)

    # Vessel
    vessel = CSAD_DP_6DOF(dt=dt, method="RK4", config_file="vessel_json.json")
    vessel.reset()

    # Build the new thruster allocator (no clipping)
    allocator = CSADThrusterAllocator(time_step=dt)

    # Wave
    wave_load, wave_params = build_random_wave()

    # 4-corner path
    waypoints = [
        [3,1,0],
        [6,1,0],
        [6,4,0],
        [6,4, np.pi/4],
        [3,4, np.pi/4],
        [3,1,0],
    ]
    vessel.set_eta( three2sixDOF(np.array(waypoints[0])) )

    # Tolerances
    dist_tol = 0.1
    head_tol = 5.0*np.pi/180
    wp_idx   = 1

    # PD gains
    kp, kd = 10.0, 0.1

    # We'll hold dr=0, ddr=0
    dr  = np.zeros(3)
    ddr = np.zeros(3)

    # Logging
    time_log    = []
    eta_log     = []
    nu_log      = []
    tau_des_log = []
    thr_cmd_log = []
    tau_act_log = []
    wave_log    = []
    ref_log     = []

    for step in range(n_steps):
        t = step*dt

        # --- Check WP
        r_cur = np.array(waypoints[wp_idx])
        eta   = vessel.get_eta()
        nPos, ePos, yaw = eta[0], eta[1], eta[5]

        dx= r_cur[0]-nPos
        dy= r_cur[1]-ePos
        d2= np.hypot(dx,dy)
        dYaw= (r_cur[2]-yaw + np.pi)%(2*np.pi)-np.pi
        if d2<dist_tol and abs(dYaw)<head_tol:
            wp_idx+=1
            if wp_idx>=len(waypoints):
                print("Reached final corner!")
                # break if you want
                wp_idx= len(waypoints)-1
            r_cur= np.array(waypoints[wp_idx])

        # --- PD => tau_des in [Fx,Fy,Mz]
        nu = vessel.get_nu()
        tau_des = feedback_linearizing_pd_3dof(vessel, eta, nu, r_cur, dr, ddr, kp, kd)

        # --- Thruster allocation => get actual [Fx,Fy,Mz]
        tau_act = allocator.allocate(fx=tau_des[0], fy=tau_des[1], mz=tau_des[2])

        # --- Wave
        tau_wave = wave_load(t, eta)

        # Combine => 6D
        tau_6= np.zeros(6)
        tau_6[[0,1,5]] = tau_act
        tau_6         += tau_wave

        # Integrate
        vessel.integrate(0.0,0.0,tau_6)

        # Log
        time_log.append(t)
        eta_log.append(vessel.get_eta().copy())
        nu_log.append(vessel.get_nu().copy())
        tau_des_log.append(tau_des.copy())
        thr_cmd_log.append(  # gather thruster-level data if you like
            [thr.force.copy() for thr in allocator.thrusters]
        )
        tau_act_log.append(tau_act.copy())
        wave_log.append(tau_wave.copy())
        ref_log.append(r_cur.copy())

    # Convert to arrays
    time_log    = np.array(time_log)
    eta_log     = np.array(eta_log)
    nu_log      = np.array(nu_log)
    tau_des_log = np.array(tau_des_log)
    tau_act_log = np.array(tau_act_log)
    wave_log    = np.array(wave_log)
    ref_log     = np.array(ref_log)

    # NOTE: thr_cmd_log is a list of lists of arrays. You can leave it as-is or
    #       transform it further (e.g. storing each thruster’s forces in shape
    #       (n_steps, n_thrusters, 2)).
    #       For demonstration, we keep it in raw list form.

    data_dict = {
        "time":      time_log,
        "eta":       eta_log,
        "nu":        nu_log,
        "tau_des":   tau_des_log,   # from PD
        "tau_act":   tau_act_log,   # actual 3D from thrusters
        "thruster_forces": thr_cmd_log,  # a more granular record
        "wave":      wave_log,
        "r":         ref_log,
        "wave_params": wave_params,
        "waypoints": waypoints,
        "kp": kp, "kd": kd,
    }

    out_file = "training_data.pkl"
    with open(out_file,"wb") as f:
        pickle.dump(data_dict, f)

    print(f"Saved data to '{out_file}'. Final boat state => {eta_log[-1]}.")

if __name__=="__main__":
    main_generate_data()
