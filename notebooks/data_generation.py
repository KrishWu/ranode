"""
Gravitational Wave Signal Generator

Generates synthetic gravitational wave signals and noise for classification experiments.
"""

import numpy as np
from typing import Tuple, Optional


def generate_noise(
    n_samples: int,
    n_timesteps: int = 300,
    noise_std: float = 1.0,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Generate Gaussian noise samples.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    n_timesteps : int
        Number of time steps per sample (default: 300 for 3s at 100Hz)
    noise_std : float
        Standard deviation of the Gaussian noise
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, n_timesteps) containing noise
    """
    if random_state is not None:
        np.random.seed(random_state)

    return np.random.normal(0, noise_std, size=(n_samples, n_timesteps))


def generate_gw_signal(
    n_samples: int,
    n_timesteps: int = 300,
    sample_rate: float = 100.0,
    f_start: float = 10.0,
    f_end: float = 40.0,
    amplitude: float = 1.0,
    chirp_duration: float = 1.0,
    random_position: bool = True,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Generate gravitational wave-like chirp signals (simplified model).

    This generates a simplified chirp signal that mimics the frequency evolution
    of a compact binary coalescence. The frequency increases from f_start to f_end.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    n_timesteps : int
        Number of time steps per sample
    sample_rate : float
        Sampling rate in Hz
    f_start : float
        Starting frequency of the chirp in Hz
    f_end : float
        Ending frequency of the chirp in Hz
    amplitude : float
        Peak amplitude of the signal
    chirp_duration : float
        Duration of the chirp in seconds
    random_position : bool
        If True, randomly position the chirp within the time window
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, n_timesteps) containing GW signals
    """
    if random_state is not None:
        np.random.seed(random_state)

    duration = n_timesteps / sample_rate
    t = np.linspace(0, duration, n_timesteps)

    signals = np.zeros((n_samples, n_timesteps))

    chirp_samples = int(chirp_duration * sample_rate)

    for i in range(n_samples):
        # Create the chirp signal
        t_chirp = np.linspace(0, chirp_duration, chirp_samples)

        # Exponential amplitude envelope: increases during inspiral
        # A(t) = max_amplitude * (t/duration)^2 - quadratic growth
        amplitude_envelope = amplitude * (t_chirp / chirp_duration) ** 2

        # Linear frequency sweep: f(t) = f0 + (f1 - f0) * t / duration
        phase = (
            2
            * np.pi
            * (
                f_start * t_chirp
                + (f_end - f_start) * t_chirp**2 / (2 * chirp_duration)
            )
        )

        # Generate the chirp
        chirp = amplitude_envelope * np.sin(phase)

        # Position the chirp in the time window
        if random_position:
            max_start = n_timesteps - chirp_samples
            if max_start > 0:
                start_idx = np.random.randint(0, max_start)
            else:
                start_idx = 0
        else:
            # Center the chirp
            start_idx = (n_timesteps - chirp_samples) // 2

        end_idx = min(start_idx + chirp_samples, n_timesteps)
        actual_length = end_idx - start_idx
        signals[i, start_idx:end_idx] = chirp[:actual_length]

    return signals


def generate_dataset(
    n_samples: int,
    n_timesteps: int = 300,
    sample_rate: float = 100.0,
    noise_std: float = 1.0,
    signal_amplitude: float = 0.5,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a complete dataset with noise-only and signal+noise samples.

    Parameters
    ----------
    n_samples : int
        Number of samples per class (total samples = 2 * n_samples)
    n_timesteps : int
        Number of time steps per sample
    sample_rate : float
        Sampling rate in Hz
    noise_std : float
        Standard deviation of the Gaussian noise
    signal_amplitude : float
        Amplitude of the GW signal relative to noise
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    noise_only : np.ndarray
        Shape (n_samples, n_timesteps) - pure noise samples
    signal_only : np.ndarray
        Shape (n_samples, n_timesteps) - pure signal (no noise)
    signal_plus_noise : np.ndarray
        Shape (n_samples, n_timesteps) - signal + noise samples
    labels : np.ndarray
        Shape (2 * n_samples,) - labels (0 for noise, 1 for signal)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate noise-only samples
    noise_only = generate_noise(n_samples, n_timesteps, noise_std)

    # Generate signal samples
    signal_only = generate_gw_signal(
        n_samples, n_timesteps, sample_rate, amplitude=signal_amplitude
    )

    # Generate noise for signal samples
    signal_noise = generate_noise(n_samples, n_timesteps, noise_std)

    # Combine signal and noise
    signal_plus_noise = signal_only + signal_noise

    # Create labels
    labels = np.concatenate(
        [np.zeros(n_samples), np.ones(n_samples)]  # noise-only  # signal+noise
    )

    return noise_only, signal_only, signal_plus_noise, labels
