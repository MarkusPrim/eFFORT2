from effort2.math.integrate import quad
from uncertainties import unumpy as unp
import numba as nb
import numpy as np


def diLog(z):
    return -1 * quad(lambda t: unp.log(1-t)/t, 0, z)[0]


def spence(z):
    return -diLog(1 - z)


@nb.njit(cache=True)
def z(w: float) -> float:
    """Variable for the expansion used in BGL and CLN.

    Args:
        w (float): [description]

    Returns:
        float: [description]
    """
    term1 = np.sqrt(w + 1)
    term2 = np.sqrt(2)
    return (term1 - term2) / (term1 + term2)