import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

# -----------------------------
# 1) Model definition
# -----------------------------
class DisturbancePredictor(nn.Module):
    """Neural network that maps [eta, nu, wave_cond] -> predicted disturbance torque (3D)."""
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(32)(x)
        x = nn.tanh(x)
        # Final layer: dimension 3 => torque disturbance
        x = nn.Dense(3)(x)
        return x


# -----------------------------
# 2) Meta-Learner 
# -----------------------------
class MetaLearner:
    def __init__(self, 
                 dt=0.1, 
                 num_models=5,
                 key_seed=42):
        """
        dt        : time step
        num_models: how many networks in the ensemble
        """
        self.dt = dt
        self.num_models = num_models
        
        # Initialize ensemble of models
        # Each model has an independent parameter set
        base_key = jax.random.PRNGKey(key_seed)
        self.models = []
        self.model_states = []
        
        for i in range(num_models):
            key_i = jax.random.split(base_key, num=i+2)[-1]
            model = DisturbancePredictor()
            params = model.init(key_i, jnp.ones(7))  # dummy input shape: (7,) => [eta(3), nu(2?), wave(2?)] adjust as needed
            # Create an optimizer for each model
            tx = optax.adam(1e-3)
            state = train_state.TrainState.create(
                apply_fn=model.apply,
                params=params,
                tx=tx
            )
            self.models.append(model)
            self.model_states.append(state)
        
        # Meta-parameters (Λ, K, Γ)
        # We'll store them in a separate dictionary
        # so we can do gradient-based updates on them as well.
        self.meta_params = {
            'Lambda': jnp.diag(jnp.array([1.0, 1.0, 0.5])),  # shape (3,3)
            'K':      jnp.diag(jnp.array([10.0, 10.0, 5.0])),# shape (3,3)
            'Gamma':  jnp.diag(jnp.array([0.1, 0.1, 0.05]))   # shape (3,3)
        }

        # Optimizer for meta-params, separate from the model(s):
        self.meta_opt = optax.adam(1e-4)
        self.meta_opt_state = self.meta_opt.init(self.meta_params)


    # -----------------------------
    # 3) Loss function 
    # -----------------------------
    def meta_loss(self, model_params_list, meta_params, batch):
        """
        model_params_list : list of parameter PyTrees (one for each ensemble member)
        meta_params       : dict with keys [Lambda, K, Gamma]
        batch             : dictionary with 'states' and 'tau_err'

          states  -> shape (batch_size, 7) or similar
          tau_err -> shape (batch_size, 3) ground truth disturbance
        """
        # We'll do an ensemble average of the predicted torque
        # from each model, then MSE vs the ground-truth disturbance.
        # For each model in the ensemble, compute:
        def single_model_pred(params, x):
            return DisturbancePredictor().apply(params, x)

        # shape (num_models, batch_size, 3)
        all_preds = []
        for i in range(self.num_models):
            # vmap over the batch dimension
            preds_i = jax.vmap(single_model_pred, in_axes=(None, 0))(model_params_list[i], batch['states'])
            all_preds.append(preds_i)
        all_preds = jnp.stack(all_preds, axis=0)  # (num_models, batch_size, 3)

        # Ensemble mean
        mean_pred = jnp.mean(all_preds, axis=0)  # (batch_size, 3)

        # MSE with ground truth disturbance
        mse_loss = jnp.mean((mean_pred - batch['tau_err']) ** 2)

        # Simple L2 regularization on meta_params:
        # Just an example to show how you might penalize large meta-params
        reg = 0.001 * (
            jnp.sum(meta_params['Lambda']**2) +
            jnp.sum(meta_params['K']**2) +
            jnp.sum(meta_params['Gamma']**2)
        )

        return mse_loss + reg


    # -----------------------------
    # 4) Single meta-training step
    # -----------------------------
    @jax.jit
    def train_step(self, model_states, meta_params, meta_opt_state, batch):
        """
        Perform a single gradient update for BOTH:
          - Each model in the ensemble
          - The meta parameters (Lambda, K, Gamma)
        """
        # 1) Compute combined loss
        def combined_loss(all_params):
            """
            all_params is a (model_params_list, meta_params) tuple
            where model_params_list = [model_0_params, ..., model_N_params]
            and meta_params is a dict {Lambda, K, Gamma}.
            """
            model_params_list, mparams = all_params
            return self.meta_loss(model_params_list, mparams, batch)

        # Combine all model params + meta_params into a single “tree”
        model_params_list = [st.params for st in model_states]
        all_params = (model_params_list, meta_params)

        # 2) Get gradients
        grads = jax.grad(combined_loss)(all_params)

        # grads is a tuple of (list_of_model_grads, meta_params_grad)
        model_grads_list, meta_grads = grads

        # 3) Update each model’s train_state
        new_model_states = []
        for st, g in zip(model_states, model_grads_list):
            new_st = st.apply_gradients(grads=g)
            new_model_states.append(new_st)

        # 4) Update meta_params
        updates, new_meta_opt_state = self.meta_opt.update(meta_grads, meta_opt_state, meta_params)
        new_meta_params = optax.apply_updates(meta_params, updates)

        return new_model_states, new_meta_params, new_meta_opt_state


    # -----------------------------
    # 5) External “public” method
    # -----------------------------
    def train_on_batch(self, batch):
        """
        Convenience method that calls `train_step` once
        and updates self.model_states, self.meta_params, self.meta_opt_state
        """
        self.model_states, self.meta_params, self.meta_opt_state = self.train_step(
            self.model_states, 
            self.meta_params, 
            self.meta_opt_state, 
            batch
        )
