import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

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

def animate_spring_system(
    u: np.ndarray,
    y: np.ndarray,
    mass_width=0.0016,
    mass_height=0.1,
    spring_height=0.05,
    num_coils=15,
    wall_x=0,
    time_s=5,
):
    """
    Animates a spring-mass system with a wall, given the force u and position y.\\
    Adapted from: https://srush.github.io/annotated-s4/
    """

    L = len(u)
    ks = np.arange(L)

    # Plot Settings
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12, 8))
    camera = Camera(fig)
    ax1.set_title("Force $u_k$")
    ax2.set_title("Position $y_k$")
    ax3.set_title("Object")
    ax1.set_xticks([], [])
    ax2.set_xticks([], [])
    ax3.set_xticks([], [])
    ax3.set_xlim(-0.4 * max(y), 1.4 * max(y))
    ax3.set_ylim(-0.2, 0.2)
    fig.tight_layout()

    # Animate plot over time
    for k in range(0, L):
        # Plot applied force
        ax1.plot(ks[:k], u[:k], color="red")

        # Plot object position
        ax2.plot(ks[:k], y[:k], color="blue")

        # Plot wall
        ax3.plot([wall_x, wall_x], [-0.2, 0.2], color='black', linewidth=2)

        # Plot spring
        spring_x = np.linspace(wall_x, y[k], num=500)
        spring_y = (spring_height / 2) * np.sin(2 * np.pi * num_coils * np.linspace(0, 1, len(spring_x)))
        ax3.plot(spring_x, spring_y, color="gray", linewidth=2)

        # Plot Mass
        mass = plt.Rectangle((y[k], -mass_height / 2),
                            mass_width, mass_height,
                            fc="steelblue", ec="black")
        ax3.add_patch(mass)

        camera.snap()

    interval = int(time_s * 1000 / L)
    plt.close()
    return camera.animate(interval=interval)