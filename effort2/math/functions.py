from effort2.math.integrate import quad
from uncertainties import unumpy as unp


def diLog(z):
    return -1 * quad(lambda t: unp.log(1-t)/t, 0, z)[0]

def spence(z):
    return -diLog(1 - z)
