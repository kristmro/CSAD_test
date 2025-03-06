import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

def truncated_normal_samples(mean_array, lower, upper, std=1.0):
    """
    Vectorized function to sample from a truncated normal distribution
    for each element in mean_array.
    
    :param mean_array: 1D numpy array of means (one for each sample)
    :param lower: Lower bound of truncation
    :param upper: Upper bound of truncation
    :param std:   Standard deviation for the underlying normal
    :return:      1D numpy array of samples with the same shape as mean_array
    """
    # Each mean might have a different alpha/beta depending on the mean
    alpha = (lower - mean_array) / std
    beta  = (upper - mean_array) / std
    
    # rvs can take array-like alpha and beta if they’re all the same shape
    samples = truncnorm.rvs(a=alpha, b=beta, loc=mean_array, scale=std, size=len(mean_array))
    return samples

def plot_wave_sampling_distribution_trunc(num_samples=10_000):
    # Wave direction: uniform between 0 and 90
    wave_direction = np.random.uniform(0, 90, num_samples)
    
    # Wave height (Hs): uniform between 0 and 5
    wave_height = np.random.uniform(0, 5, num_samples)
    
    # Mean wave period is linearly interpolated between 6 s (at Hs=0) and 20 s (at Hs=5)
    period_mean = 6 + (wave_height / 5.0) * (20 - 6)
    
    # Instead of sampling from a normal and clipping,
    # sample from a truncated normal in [6, 20]
    wave_period = truncated_normal_samples(mean_array=period_mean,
                                           lower=6, upper=20,
                                           std=1.0)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # (A) Wave Direction histogram
    axes[0,0].hist(wave_direction, bins=30, density=True, alpha=0.7, color='blue')
    axes[0,0].set_xlabel('Wave Direction [deg]')
    axes[0,0].set_ylabel('Density')
    axes[0,0].set_title('Distribution of Wave Direction')
    
    # (B) Wave Height histogram
    axes[0,1].hist(wave_height, bins=30, density=True, alpha=0.7, color='orange')
    axes[0,1].set_xlabel('Wave Height [m]')
    axes[0,1].set_ylabel('Density')
    axes[0,1].set_title('Distribution of Wave Height')
    
    # (C) Wave Period histogram
    axes[1,0].hist(wave_period, bins=30, density=True, alpha=0.7, color='green')
    axes[1,0].set_xlabel('Wave Period [s]')
    axes[1,0].set_ylabel('Density')
    axes[1,0].set_title('Distribution of Wave Period')
    
    # (D) 2D histogram: Wave Height vs. Wave Period
    h = axes[1,1].hist2d(
        wave_height, wave_period, 
        bins=(30, 30),
        range=[[0, 5], [6, 20]], 
        cmap='viridis', 
        density=True
    )
    plt.colorbar(h[3], ax=axes[1,1], label='Density')
    axes[1,1].set_xlabel('Wave Height [m]')
    axes[1,1].set_ylabel('Wave Period [s]')
    axes[1,1].set_title('Joint Distribution of Wave Height & Period')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_wave_sampling_distribution_trunc(num_samples=10_000)
