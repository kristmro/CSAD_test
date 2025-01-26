import jax
import jax.numpy as jnp

class MetaAdaptiveController:
    def __init__(self, meta_learner, dt=0.1):
        """
        meta_learner: instance of the MetaLearner class defined above
        dt          : time step
        """
        self.meta_learner = meta_learner  # needed to get ensemble & meta_params
        self.dt = dt

        # Adaptive disturbance estimate (for each DOF)
        self.dist_est = jnp.zeros(3)

        # Control limits
        self.tau_max = jnp.array([1000., 1000., 500.])

    def compute_action(self, state, goal):
        """
        state: dict with boat_position, boat_orientation, velocities, wave_conditions
        goal : desired (x, y) or similar
        Returns: jnp.array of shape (3,) => the commanded tau
        """
        # Convert state to JAX array
        eta = jnp.array([
            state["boat_position"][0],
            state["boat_position"][1],
            state["boat_orientation"]
        ])
        # For simplicity, assume velocity is 2D plus yaw rate => shape (3,).
        nu = jnp.array(state["velocities"])
        
        # wave_cond can be 1D, shape something like (2,) or (3,).
        wave_cond = jnp.array(state["wave_conditions"])

        # Disturbance prediction via ensemble
        tau_wave_hat = self._ensemble_predict(eta, nu, wave_cond)

        # Retrieve meta-parameters 
        Lambda = self.meta_learner.meta_params['Lambda']
        K      = self.meta_learner.meta_params['K']
        Gamma  = self.meta_learner.meta_params['Gamma']

        # Error signals
        eta_d, nu_d, dnu_d = self._reference_generator(eta, goal)
        e = eta - eta_d
        edot = nu - nu_d

        # Sliding surface
        s = edot + Lambda @ e

        # Online update for your “theta” (if you want a direct param adaptation).
        # Or you can skip if the network is your disturbance predictor.
        # Example: self.dist_est = self.dist_est + Gamma @ s * dt
        # But typically the neural net is handling that, so you might or might not need this.

        # Control law (classic sliding approach)
        # tau = K * s + tau_wave_hat - Lambda( nu - nu_d )  (example)
        tau = (K @ s) + tau_wave_hat - Lambda @ (nu - nu_d)

        # Clip
        return jnp.clip(tau, -self.tau_max, self.tau_max)

    def _ensemble_predict(self, eta, nu, wave_cond):
        """
        Averages predictions from all models in meta_learner
        """
        # Build the input vector
        # shape = 7 if you do (eta(3), nu(3), wave_cond(1)) or something similar
        # Adjust as needed
        x = jnp.concatenate([eta, nu, wave_cond])

        # For each model, compute disturbance
        preds = []
        for i, state in enumerate(self.meta_learner.model_states):
            pred_i = self.meta_learner.models[i].apply(state.params, x)
            preds.append(pred_i)
        preds = jnp.stack(preds, axis=0)  # (num_models, 3)

        # Ensemble mean
        return jnp.mean(preds, axis=0)

    def _reference_generator(self, eta, goal):
        """
        Example of a simple PD reference:
          - goal[0], goal[1] => desired x,y
          - heading = arctan2(y - y_current, x - x_current)
        """
        kp = 0.2
        kd = 0.1
        desired_heading = jnp.arctan2(goal[1] - eta[1], goal[0] - eta[0])
        eta_d = jnp.array([goal[0], goal[1], desired_heading])

        nu_d = kp * (eta_d - eta)
        dnu_d = -kd * nu_d

        return eta_d, nu_d, dnu_d
