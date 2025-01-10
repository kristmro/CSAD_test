import numpy as np
import torch
from torch import optim
from copy import deepcopy

# Reptile Meta-Learner class
class ReptileMetaLearner:
    def __init__(self, model, alpha=0.1, inner_lr=0.01, meta_steps=5):
        """
        Args:
        - model: The neural network model (controller) to be trained.
        - alpha: Step size for meta-update.
        - inner_lr: Learning rate for task-specific adaptation.
        - meta_steps: Number of tasks to sample per meta-iteration.
        """
        self.model = model
        self.alpha = alpha
        self.inner_lr = inner_lr
        self.meta_steps = meta_steps
    
    def adapt_to_task(self, task_env, epochs=10, evaluate=False):
        """
        Adapts the model to a single task environment.
        
        Args:
        - task_env: The environment to train the model on.
        - epochs: Number of epochs for adaptation to each task.
        - evaluate: If True, only evaluates without updating weights.
        
        Returns:
        - The adapted model's weights.
        """
        task_model = deepcopy(self.model)
        optimizer = optim.Adam(task_model.parameters(), lr=self.inner_lr)

        for epoch in range(epochs):
            state = task_env.reset()
            done = False
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action = task_model(state_tensor)
                
                next_state, reward, done, _ = task_env.step(action.detach().numpy())
                loss = -reward
                
                if not evaluate:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                state = next_state

        return task_model.state_dict()
    
    def meta_train(self, task_envs):
        """
        Meta-trains the model over multiple tasks using Reptile.
        
        Args:
        - task_envs: List of task-specific environments for meta-training.
        
        Returns:
        - The final meta-trained weights.
        """
        initial_weights = deepcopy(self.model.state_dict())

        for step in range(self.meta_steps):
            task_env = np.random.choice(task_envs)

            # Adapt the model to the sampled task
            task_weights = self.adapt_to_task(task_env)

            # Reptile meta-update
            for param in initial_weights.keys():
                initial_weights[param] += self.alpha * (task_weights[param] - initial_weights[param])

            # Update the model with the new meta-optimized weights
            self.model.load_state_dict(initial_weights)

        print("Meta-training complete")
        return initial_weights
