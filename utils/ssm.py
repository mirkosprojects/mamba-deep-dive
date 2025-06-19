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