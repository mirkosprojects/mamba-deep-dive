import numpy as np

def discretize(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts continuous-time matrices A, B into discrete-time versions using the bilinear transform.

    Parameters:
        A (ndarray): continuous-time state transition matrix
        B (ndarray): continuous-time input matrix
        dt (float): time step for discretization
    
    Returns:
        (A_d, B_d) (ndarray, ndarray): discrete-time state transition matrix A_d and discrete-time input matrix B_d
    """
    I = np.eye(A.shape[0])
    A_d = np.linalg.inv(I - 0.5 * A * dt) @ (I + 0.5 * A * dt)
    B_d = np.linalg.inv(I - 0.5 * A * dt) @ (B * dt)

    # Alternatively, using solve_triangular for better numerical stability and faster computation
    # A_d = la.solve_triangular(I - 0.5 * A * dt, I + 0.5 * A * dt, lower=True)
    # B_d = la.solve_triangular(I - 0.5 * A * dt, B * dt, lower=True)
    return A_d, B_d

def whitesignal(period, dt, freq, rms=0.5):
    """
    Produces output signal of length period / dt, band-limited to frequency freq\\
    Adapted from the nengo library
    """

    if freq is not None and freq < 1. / period:
        raise ValueError(f"Make ``{freq=} >= 1. / {period=}`` to produce a non-zero signal",)

    nyquist_cutoff = 0.5 / dt
    if freq > nyquist_cutoff:
        raise ValueError(f"{freq} must not exceed the Nyquist frequency for the given dt ({nyquist_cutoff:0.3f})")

    n_coefficients = int(np.ceil(period / dt / 2.))
    shape = (n_coefficients + 1,)
    sigma = rms * np.sqrt(0.5)
    coefficients = 1j * np.random.normal(0., sigma, size=shape)
    coefficients[..., -1] = 0.
    coefficients += np.random.normal(0., sigma, size=shape)
    coefficients[..., 0] = 0.

    set_to_zero = np.fft.rfftfreq(2 * n_coefficients, d=dt) > freq
    coefficients *= (1-set_to_zero)
    power_correction = np.sqrt(1. - np.sum(set_to_zero, dtype=float) / n_coefficients)
    if power_correction > 0.: coefficients /= power_correction
    coefficients *= np.sqrt(2 * n_coefficients)
    signal = np.fft.irfft(coefficients, axis=-1)
    signal = signal - signal[..., :1]  # Start from 0
    return signal