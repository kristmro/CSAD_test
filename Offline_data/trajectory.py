import jax
import jax.numpy as jnp
import pickle
from CSADtesting.filters.reference_filter import ThrdOrderRefFilter

class TrajectoryGenerator:
    def __init__(self, simTime, dt, initial_eta=jnp.array([2, 2, 0])):
        self.simTime = simTime
        self.dt = dt
        # Ensure initial_eta is a JAX array
        if not isinstance(initial_eta, jnp.ndarray):
            self.initial_eta = jnp.array(initial_eta)
        else:
            self.initial_eta = initial_eta

        self.t = jnp.arange(0, simTime, dt)
        
        self.set_points = [
            jnp.array([2.0, 2.0, 0.0]),
            jnp.array([4.0, 2.0, 0.0]),
            jnp.array([4.0, 4.0, 0.0]),
            jnp.array([4.0, 4.0, -jnp.pi/4]),
            jnp.array([2.0, 4.0, -jnp.pi/4]),
            jnp.array([2.0, 2.0, 0.0])
        ]
        
        # The reference filter is NumPy based.
        # Pass initial_eta as a list so that np.array(initial_eta) works correctly.
        self.ref_model = ThrdOrderRefFilter(dt, initial_eta=self.initial_eta.tolist())
        self.store_xd = jnp.zeros((len(self.t), 9))

    def get_four_corner_nd(self, step_count, start_eta):
        # Ensure start_eta is a JAX array.
        start_eta = jnp.array(start_eta)
        current_time = self.t[step_count]
        
        # Decide which setpoint index to use
        if jnp.allclose(start_eta, self.set_points[0], atol=1e-3):
            if current_time < 10.0:
                idx = 0
            else:
                shifted_time = current_time - 10.0
                remaining_time = self.simTime - 10.0
                segment_duration = remaining_time / 5.0
                idx = 1 + min(4, int(shifted_time // segment_duration))
        else:
            if current_time > 5 * self.simTime / 6:
                idx = 5
            elif current_time > 4 * self.simTime / 6:
                idx = 4
            elif current_time > 3 * self.simTime / 6:
                idx = 3
            elif current_time > 2 * self.simTime / 6:
                idx = 2
            elif current_time > self.simTime / 6:
                idx = 1
            else:
                idx = 0

        # Pass the setpoint as a list to the NumPy filter.
        self.ref_model.set_eta_r(self.set_points[idx].tolist())
        self.ref_model.update()
        
        # Convert the filter outputs (which are NumPy arrays) to JAX arrays.
        eta_d = jnp.array(self.ref_model.get_eta_d())
        eta_d_dot = jnp.array(self.ref_model.get_eta_d_dot())
        eta_d_ddot = jnp.array(self.ref_model.get_eta_d_ddot())
        nu_d_body = jnp.array(self.ref_model.get_nu_d())
        
        # Convert the internal state to a JAX array before storing.
        self.store_xd = self.store_xd.at[step_count].set(jnp.array(self.ref_model._x))
        
        return eta_d, eta_d_dot, eta_d_ddot, nu_d_body

    def generate_trajectory(self, save):
        """Generate full trajectory and optionally save to a pickle file."""
        trajectory_data = []
        
        for i in range(len(self.t)):
            data = self.get_four_corner_nd(i, self.initial_eta)
            trajectory_data.append({
                'time': float(self.t[i]),
                'eta_d': data[0],       # desired position in NED
                'eta_d_dot': data[1],   # desired velocity in NED
                'eta_d_ddot': data[2],  # desired acceleration in NED
                'nu_d_body': data[3]    # desired velocity in body
            })
        
        # Save to pickle file if requested (convert JAX arrays to lists for pickling).
        if save:
            trajectory_data_pickle = []
            for data in trajectory_data:
                trajectory_data_pickle.append({
                    'time': data['time'],
                    'eta_d': data['eta_d'].tolist(),
                    'eta_d_dot': data['eta_d_dot'].tolist(),
                    'eta_d_ddot': data['eta_d_ddot'].tolist(),
                    'nu_d_body': data['nu_d_body'].tolist()
                })
            with open('trajectory_data_02_450_008.pkl', 'wb') as f:
                pickle.dump(trajectory_data_pickle, f)
            
        return trajectory_data

def main():
    simTime = 450
    dt = 0.08
    # Even if provided as a list, the constructor converts it to a JAX array.
    initial_eta = [2, 2, 0]
    trajectory = TrajectoryGenerator(simTime, dt, initial_eta)
    #trajectory_data = trajectory.generate_trajectory(save=True)

    # # Now, plot the trajectory data.
    # import matplotlib.pyplot as plt
    # time = [data['time'] for data in trajectory_data]
    # eta_d = [data['eta_d'] for data in trajectory_data]
    # eta_d_dot = [data['eta_d_dot'] for data in trajectory_data]
    # eta_d_ddot = [data['eta_d_ddot'] for data in trajectory_data]
    # nu_d_body = [data['nu_d_body'] for data in trajectory_data]
    
    # # Stack the lists to form matrices for easier slicing.
    # eta_d = jnp.stack(eta_d)       # Shape: (num_timesteps, 3)
    # eta_d_dot = jnp.stack(eta_d_dot)
    # eta_d_ddot = jnp.stack(eta_d_ddot)
    # nu_d_body = jnp.stack(nu_d_body)
    
    # plt.figure()
    # plt.plot(time, eta_d[:, 0], label='Surge')
    # plt.plot(time, eta_d[:, 1], label='Sway')
    # plt.plot(time, eta_d[:, 2], label='Yaw')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Position (m)')
    # plt.title('Desired Position in NED')
    # plt.legend()
    # plt.show()
    
    # plt.figure()
    # plt.plot(time, eta_d_dot[:, 0], label='Surge')
    # plt.plot(time, eta_d_dot[:, 1], label='Sway')
    # plt.plot(time, eta_d_dot[:, 2], label='Yaw')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Velocity (m/s)')
    # plt.title('Desired Velocity in NED')
    # plt.legend()
    # plt.show()
    
    # plt.figure()
    # plt.plot(time, eta_d_ddot[:, 0], label='Surge')
    # plt.plot(time, eta_d_ddot[:, 1], label='Sway')
    # plt.plot(time, eta_d_ddot[:, 2], label='Yaw')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Acceleration (m/s²)')
    # plt.title('Desired Acceleration in NED')
    # plt.legend()
    # plt.show()
    
    # plt.figure()
    # plt.plot(time, nu_d_body[:, 0], label='Surge')
    # plt.plot(time, nu_d_body[:, 1], label='Sway')
    # plt.plot(time, nu_d_body[:, 2], label='Yaw')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Velocity (m/s)')
    # plt.title('Desired Velocity in Body')
    # plt.legend()
    # plt.show()
    
    # plt.figure()
    # plt.plot(eta_d[:, 1], eta_d[:, 0])
    # plt.xlabel('Sway (m)')
    # plt.ylabel('Surge (m)')
    # plt.title('Desired Position in NED')
    # plt.show()
    
    # plt.figure()
    # plt.plot(eta_d_dot[:, 0], nu_d_body[:, 0], label='Surge')
    # plt.plot(eta_d_dot[:, 1], nu_d_body[:, 1], label='Sway')
    # plt.plot(eta_d_dot[:, 2], nu_d_body[:, 2], label='Yaw')
    # plt.xlabel('Velocity in NED (m/s)')
    # plt.ylabel('Velocity in Body (m/s)')
    # plt.title('Desired Velocity')
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    main()
