import tensorflow as tf
from tensorflow.keras import layers, optimizers
import numpy as np
from MCSimPython.utils import three2sixDOF
from scipy.optimize import minimize

class DynamicsModel(nn.Module):
    """Neural network for learning system dynamics."""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DynamicsModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state, action):
        """
        Forward pass of the dynamics model.
        state: Current state tensor (batch_size, state_dim).
        action: Action tensor (batch_size, action_dim).
        """
        # Ensure action has the same dimensions as state
        if action.ndim == 1:  # If action is 1D, unsqueeze to make it 2D
            action = action.unsqueeze(0)

        # Concatenate state and action
        input_data = torch.cat([state, action], dim=-1)
        return self.net(input_data)



class MBRLController:
    """Model-Based Reinforcement Learning Controller."""
    def __init__(self, state_dim, action_dim, hidden_dim=128, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dynamics_model = DynamicsModel(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.dynamics_model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.state_dim = state_dim
        self.action_dim = action_dim

    def train_dynamics_model(self, optimizer, loss_fn, data_loader, epochs=50):
        """Train the dynamics model."""
        self.dynamics_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for states, actions, next_states in data_loader:
                states = states.to(self.device)
                actions = actions.to(self.device)
                next_states = next_states.to(self.device)

                # Predict next states
                predicted_next_states = self.dynamics_model(states, actions)

                # Compute loss
                loss = loss_fn(predicted_next_states, next_states)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader):.6f}")



    def plan_action_mpc(self, state, trajectory, step_index, horizon=5, lambda_reg=10):
        """
        Plan actions using Model Predictive Control (MPC).
        state: Current vessel state (tensor).
        trajectory: List of target positions in NED coordinates.
        step_index: Current index in the trajectory.
        horizon: Prediction horizon for MPC.
        lambda_reg: Regularization weight for action penalties.
        """
        # Stop if at the last point of the trajectory
        if step_index >= len(trajectory):
            return torch.zeros((1, self.action_dim)).to(self.device), step_index  # Stop action

        # Extract the next `horizon` target positions (wrap if necessary)
        target_positions = trajectory[step_index:min(step_index + horizon, len(trajectory))]

        # Define the cost function for optimization
        def cost_function(actions):
            total_cost = 0.0
            predicted_state = state.clone()

            for i in range(len(target_positions)):
                # Extract action for this step
                action = torch.tensor(actions[i * self.action_dim:(i + 1) * self.action_dim], dtype=torch.float32).to(self.device)

                if action.ndim == 1:
                    action = action.unsqueeze(0)

                # Predict next state using the dynamics model
                predicted_state = self.dynamics_model(predicted_state, action)

                # Compute error to the target position
                target_position = torch.tensor(target_positions[i], dtype=torch.float32).to(self.device)
                error = torch.norm(predicted_state[0, :2] - target_position[:2])  # Only North and East
                total_cost += error**2 + lambda_reg * torch.norm(action)**2  # Add action penalty

            return total_cost.item()

        # Initial guess for actions (zeros)
        initial_guess = np.zeros(len(target_positions) * self.action_dim)

        # Optimize the action sequence
        result = minimize(cost_function, initial_guess, method='L-BFGS-B', options={'maxiter': 100})

        # Extract the first action from the optimized sequence
        optimal_actions = result.x.reshape(len(target_positions), self.action_dim)
        first_action = optimal_actions[0]

        # Update the step index
        step_index += 1
        return torch.tensor(first_action, dtype=torch.float32).unsqueeze(0).to(self.device), step_index


    def collect_data(self, environment, reference_path, steps_per_episode=500, proportional_gain=0.1):
        """
        Collect training data by guiding the vessel along the smooth reference path.
        environment: The simulation environment.
        reference_path: Smooth reference trajectory (Nx3 array of positions: [N, E, ψ]).
        steps_per_episode: Number of steps per episode.
        proportional_gain: Gain for proportional control during training.
        """
        data = []
        step_index = 0  # Start at the first point of the reference path

        environment.reset()
        state = torch.tensor(environment.get_state()[:self.state_dim], dtype=torch.float32).unsqueeze(0).to(self.device)

        for step in range(steps_per_episode):
            # Get the current target position from the reference path
            target_position = torch.tensor(reference_path[step_index], dtype=torch.float32).to(self.device)

            # Compute the error in NED frame (North, East, Heading)
            error = target_position - state[0]  # Target - Current position

            # Compute action as proportional control
            action = torch.zeros((1, self.action_dim)).to(self.device)
            action[0, :3] = error * proportional_gain  # Control surge, sway, and yaw

            # Apply the action to the environment
            next_state = environment.step(action.cpu().numpy())
            next_state = torch.tensor(next_state[:self.state_dim], dtype=torch.float32).unsqueeze(0).to(self.device)

            # Collect data
            data.append((state.squeeze(0).cpu().numpy(), action.cpu().numpy().squeeze(0), next_state.squeeze(0).cpu().numpy()))
            state = next_state

            # Update step index to the next point in the reference path
            step_index = (step_index + 1) % len(reference_path)

        # Convert collected data to tensors
        states, actions, next_states = zip(*data)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        return states, actions, next_states

