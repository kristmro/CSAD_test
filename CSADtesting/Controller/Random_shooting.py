import numpy as np

class RSBoatOptimizer:
    def __init__(self, config):
        """
        Random Shooting Optimizer for boat control.
        
        Args:
            config (dict): Configuration dictionary with keys:
                - max_iters: Maximum number of iterations (default: 5).
                - lb: Lower bound of actions (default: -1.0).
                - ub: Upper bound of actions (default: 1.0).
                - popsize: Population size for random samples (default: 500).
                - sol_dim: Dimensionality of the solution (action_dim * horizon).
                - cost_fn: Cost function to evaluate action sequences.
        """
        self.max_iters = config["max_iters"]  # Maximum number of iterations
        self.lb = config["lb"]  # Lower bound for actions
        self.ub = config["ub"]  # Upper bound for actions
        self.popsize = config["popsize"]  # Number of random samples
        self.sol_dim = config["sol_dim"]  # Dimensionality of solution (action_dim * horizon)
        self.cost_function = config["cost_fn"]  # Cost function for evaluation

    def obtain_solution(self, init_mean=None, init_var=None):
        """
        Finds the best action sequence using Random Shooting.
        
        Args:
            init_mean (np.ndarray): Initial mean of the candidate distribution (optional).
            init_var (np.ndarray): Initial variance of the candidate distribution (optional).
        
        Returns:
            np.ndarray: The best action sequence found.
        """
        if init_mean is None or init_var is None:
            # Uniform random sampling
            samples = np.random.uniform(self.lb, self.ub, size=(self.popsize, self.sol_dim))
        else:
            # Gaussian random sampling
            samples = np.random.normal(init_mean, init_var, size=(self.popsize, self.sol_dim))
            samples = np.clip(samples, self.lb, self.ub)  # Ensure samples are within bounds

        # Evaluate costs for all sampled action sequences
        costs = self.cost_function(samples)

        # Select and return the action sequence with the minimum cost
        return samples[np.argmin(costs)]
