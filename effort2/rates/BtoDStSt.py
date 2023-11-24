# TO DO: add the rates here
# dGamma/dw can be found in arxiv:1711.03110
# dGamma/dwdcosTheta can be found in arxiv:1606:09300

import numpy as np
from effort2.formfactors.kinematics import Kinematics
from effort2.formfactors.BLR import BToDStarStarBroad, BToDStarStarNarrow
from effort2.math.integrate import quad


class BtoD0St(BToDStarStarBroad):

    def __init__(
        self,
        Vcb: float,
        m_D: float,
        m_B: float,
        m_L:float=0,
        G_F: float=1.1663787e-5,
        m_c: float=1.31,
        m_b: float=4.71,
        params: tuple=(0.70, 0.2, 0.6),
        alphaS: float=0.26,
        LambdaBar: float=0.40,
        LambdaBarPrime: float=0.80,
        LambdaBarStar: float=0.76,
        chi1: float=0,
        chi2: float=0,
    ) -> None:

        super().__init__(
            m_c,
            m_b,
            params,
            alphaS,
            LambdaBar,
            LambdaBarPrime,
            LambdaBarStar,
            chi1,
            chi2,
        )
        self.m_c = m_c
        self.m_b = m_b
        self.alphaS = alphaS / np.pi
        self.LambdaBar = LambdaBar
        self.LambdaBarPrime = LambdaBarPrime
        self.LambdaBarStar = LambdaBarStar
        self.chi1 = chi1
        self.chi2 = chi2

        self.Z1, self.zetap, self.zeta1 = self.set_model_parameters(params)

        self.rm = m_D / m_B
        self.rho = (m_L / m_B)**2

        self.Gamma0 = (G_F**2 * Vcb**2 * m_B**5) / (192 * np.pi**3)

        self.kinematics = Kinematics(m_B, m_D, m_L)
        self.w_min, self.w_max = self.kinematics.w_range_numerical_stable


    def q2(
        self,
        w: float,
    ):
        rm = self.rm
        return 1 + rm**2 - 2 * rm * w

    def dGamma_dw(
        self,
        w: float,
    ) -> float:
        Gamma0 = self.Gamma0
        rho = self.rho
        rm = self.rm
        q2 = self.q2(w)
        zeta = self.Z1 * (1 + self.zetap * (w - 1)) # LO expansion of IW functions: Equation (36) in arxiv:1711.03110
        gminus = self.gminus(w) * zeta
        gplus = self.gplus(w) * zeta
        return 4 * Gamma0 * rm**3 * np.sqrt(w**2 - 1) * (q2 - rho)**2 / q2**3 * (gminus**2 * (w - 1) * (rho *((1 + rm**2) * (2 * w - 1) + 2 * rm * (w - 2)) + (1 - rm)**2 * (w + 1) * q2) + gplus**2 * (w + 1) * (rho * ((1 + rm**2) * (2 * w + 1) - 2 * rm * (w + 2)) + (1 + rm)**2 * (w - 1) * q2) - 2 * gminus * gplus * (1 - rm**2) * (w**2 - 1) * (q2 + 2 * rho))

    def Gamma(
        self,
        wmin: float=None,
        wmax: float=None,
    ) -> float:
        wmin = self.w_min if wmin is None else wmin
        wmax = self.w_max if wmax is None else wmax
        assert self.w_min <= wmin < wmax <= self.w_max, f"{wmin}, {wmax}"

        return quad(lambda w: self.dGamma_dw(w), wmin, wmax)[0]


if __name__ == "__main__":
    pass
