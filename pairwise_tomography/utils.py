import numpy as np
from qiskit.tools.qi.qi import outer
import scipy.linalg as la

def concurrence(state):
    """Calculate the concurrence.

    Args:
        state (np.array): a quantum state (1x4 array) or a density matrix (4x4
                          array)
    Returns:
        float: concurrence.
    Raises:
        Exception: if attempted on more than two qubits.
    """
    rho = np.array(state)
    if rho.ndim == 1:
        rho = outer(state)
    if len(state) != 4:
        raise Exception("Concurrence is only defined for more than two qubits")

    YY = np.fliplr(np.diag([-1, 1, 1, -1]))
    A = rho.dot(YY).dot(rho.conj()).dot(YY)
    w = la.eigvals(A)
    w = np.sort(np.real(w))
    w = np.sqrt(np.maximum(w, 0))
    return max(0.0, w[-1] - np.sum(w[0:-1]))
