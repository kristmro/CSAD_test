import numpy as np
import sys
sys.path.append('./CSADtesting')  # Adjust if needed

from Environment.GridBoatEnv import GridWaveEnvironment
from Controller.meta_learner import MetaLearner
from Controller.meta_controller import MetaAdaptiveController

def train_meta_controller():
    # Initialize environment & meta-learner
    env = GridWaveEnvironment()
    meta_learner = MetaLearner()
    controller = MetaAdaptiveController(meta_learner)
    
    # Training parameters
    num_epochs = 10
    episodes_per_epoch = 2

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for _ in range(episodes_per_epoch):
            # Randomize environment scenario
            start_pos = (np.random.uniform(0, 5),
                         np.random.uniform(0, 5),
                         np.random.uniform(0, 2*np.pi))
            wave_cond = (np.random.uniform(0.5, 2.5),
                         np.random.uniform(3, 6))
            
            env.set_task(start_pos, wave_cond)
            done = False
            states_buffer, disturbances_buffer = [], []
            
            while not done:
                state = env.get_state()
                action = controller.compute_action(state, env.goal)
                next_state, reward, done, info = env.step(action)

                # Suppose "info['true_disturbance']" holds the actual wave torque 
                # for that step (3D). Or you have some known "tau_wave" from the sim.
                true_dist = info["true_disturbance"]  

                # Store data for training
                # Flatten state into shape (7,) if your net expects that.
                # e.g. [x, y, heading, u, v, r, wave_param1, wave_param2, ...]
                # Make sure you keep consistent shapes
                s_input = np.concatenate([
                    state["boat_position"],
                    [state["boat_orientation"]],
                    state["velocities"],
                    state["wave_conditions"]
                ])
                states_buffer.append(s_input)
                disturbances_buffer.append(true_dist)
            
            # Now build a "batch" for meta_learner
            states_arr = np.array(states_buffer)         # shape (T, 7)
            dist_arr   = np.array(disturbances_buffer)   # shape (T, 3)
            batch = {
                "states": states_arr,
                "tau_err": dist_arr
            }

            # Convert to jax.numpy inside train_on_batch
            meta_learner.train_on_batch(batch)
        
        # (Optional) Evaluate or compute epoch_loss
        # Could do a quick pass of meta_loss on some validation
        # For simplicity, we skip it here
        print(f"Epoch {epoch} completed.")


if __name__ == "__main__":
    train_meta_controller()
