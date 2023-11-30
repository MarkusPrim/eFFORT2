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


class BtoD1St(BToDStarStarBroad):

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
        gV1 = self.gV1(w) * zeta
        gV2 = self.gV2(w) * zeta
        gV3 = self.gV3(w) * zeta
        gA = self.gA(w) * zeta
        return 2 * Gamma0 * rm**3 * np.sqrt(w**2 - 1) * (q2 - rho)**2 / q2**3 * (gV1**2 * (2 * q2 * ((w - rm)**2 + 2 * q2) + rho * (4 * (w - rm)**2 - q2)) + (w**2 - 1) * (gV2**2 * (2 * rm**2 * q2 * (w**2 - 1) + rho * (3 * q2 + 4 * rm**2 * (w**2 - 1))) + gV3**2 * (2 * q2 * (w**2 - 1) + rho * (4 * (w - rm)**2 - q2)) + 2 * gA**2 * q2 * (2 * q2 + rho) + 2 * gV1 * gV2 * (2 * rm * q2 * (w - rm) + rho * (3 - rm**2 - 2 * rm * w)) + 4 * gV1 * gV3 * (w - rm) * (q2 + 2 * rho) + 2 * gV2 * gV3 * (2 * rm * q2 * (w**2 - 1) + rho * (3 * w * q2 + 4 * rm * (w**2 - 1)))))

    def Gamma(
        self,
        wmin: float=None,
        wmax: float=None,
    ) -> float:
        wmin = self.w_min if wmin is None else wmin
        wmax = self.w_max if wmax is None else wmax
        assert self.w_min <= wmin < wmax <= self.w_max, f"{wmin}, {wmax}"

        return quad(lambda w: self.dGamma_dw(w), wmin, wmax)[0]
    

class BtoD1(BToDStarStarNarrow):

    def __init__(
        self,
        Vcb: float,
        m_D: float,
        m_B: float,
        m_L:float=0,
        G_F: float=1.1663787e-5,
        m_c: float=1.31,
        m_b: float=4.71,
        params: tuple=(0.70, -1.6, -0.5, 2.9),
        alphaS: float=0.26,
        LambdaBar: float=0.40,
        LambdaBarPrime: float=0.80,
        LambdaBarStar: float=0.76,
        eta1: float=0,
        eta2: float=0,
        eta3: float=0,
    ) -> None:

        super().__init__(
            m_c,
            m_b,
            params,
            alphaS,
            LambdaBar,
            LambdaBarPrime,
            LambdaBarStar,
            eta1,
            eta2,
            eta3,
        )
        self.m_c = m_c
        self.m_b = m_b
        self.alphaS = alphaS / np.pi
        self.LambdaBar = LambdaBar
        self.LambdaBarPrime = LambdaBarPrime
        self.LambdaBarStar = LambdaBarStar
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3

        self.T1, self.taup, self.tau1, self.tau2 = self.set_model_parameters(params)

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
        tau = self.T1 * (1 + self.taup * (w - 1)) # LO expansion of IW functions: Equation (36) in arxiv:1711.03110
        fV1 = self.fV1(w) * tau
        fV2 = self.fV2(w) * tau
        fV3 = self.fV3(w) * tau
        fA = self.fA(w) * tau
        return 2 * Gamma0 * rm**3 * np.sqrt(w**2 - 1) * (q2 - rho)**2 / q2**3 * (fV1**2 * (2 * q2 * ((w - rm)**2 + 2 * q2) + rho * (4 * (w - rm)**2 - q2)) + (w**2 - 1) * (fV2**2 * (2 * rm**2 * q2 * (w**2 - 1) + rho * (3 * q2 + 4 * rm**2 * (w**2 - 1))) + fV3**2 * (2 * q2 * (w**2 - 1) + rho * (4 * (w - rm)**2 - q2)) + 2 * fA**2 * q2 * (2 * q2 + rho) + 2 * fV1 * fV2 * (2 * rm * q2 * (w - rm) + rho * (3 - rm**2 - 2 * rm * w)) + 4 * fV1 * fV3 * (w - rm) * (q2 + 2 * rho) + 2 * fV2 * fV3 * (2 * rm * q2 * (w**2 - 1) + rho * (3 * w * q2 + 4 * rm * (w**2 - 1)))))

    def Gamma(
        self,
        wmin: float=None,
        wmax: float=None,
    ) -> float:
        wmin = self.w_min if wmin is None else wmin
        wmax = self.w_max if wmax is None else wmax
        assert self.w_min <= wmin < wmax <= self.w_max, f"{wmin}, {wmax}"

        return quad(lambda w: self.dGamma_dw(w), wmin, wmax)[0]
    

class BToD2St(BToDStarStarNarrow):

    def __init__(
        self,
        Vcb: float,
        m_D: float,
        m_B: float,
        m_L:float=0,
        G_F: float=1.1663787e-5,
        m_c: float=1.31,
        m_b: float=4.71,
        params: tuple=(0.70, -1.6, -0.5, 2.9),
        alphaS: float=0.26,
        LambdaBar: float=0.40,
        LambdaBarPrime: float=0.80,
        LambdaBarStar: float=0.76,
        eta1: float=0,
        eta2: float=0,
        eta3: float=0,
    ) -> None:

        super().__init__(
            m_c,
            m_b,
            params,
            alphaS,
            LambdaBar,
            LambdaBarPrime,
            LambdaBarStar,
            eta1,
            eta2,
            eta3,
        )
        self.m_c = m_c
        self.m_b = m_b
        self.alphaS = alphaS / np.pi
        self.LambdaBar = LambdaBar
        self.LambdaBarPrime = LambdaBarPrime
        self.LambdaBarStar = LambdaBarStar
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3

        self.T1, self.taup, self.tau1, self.tau2 = self.set_model_parameters(params)

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
        tau = self.T1 * (1 + self.taup * (w - 1)) # LO expansion of IW functions: Equation (36) in arxiv:1711.03110
        kA1 = self.kA1(w) * tau
        kA2 = self.kA2(w) * tau
        kA3 = self.kA3(w) * tau
        kV = self.kV(w) * tau
        return (2/3) * Gamma0 * rm**3 * (w**2 - 1)**(3/2) * (q2 - rho)**2 / q2**3 * (kA1**2 * (2 * q2 *(2 * (w - rm)**2 + 3 * q2) + rho * (8 * (w - rm)**2 - 3 * q2)) + 2 * (w**2 - 1) * (kA2**2 * (2 * rm**2 * q2 (w**2 - 1) + rho * (3 * q2 + 4 * rm**2 * (w**2 - 1))) + kA3**2 * (2 * q2 * (w**2 - 1) + rho * (4 * (w - rm)**2 - q2)) + 3 * kV**2 * q2 * (q2 + rho/2) + 2 * kA1 * kA2 * (2 * rm * q2 * (w - rm) + rho * (3 - rm**2 - 2 * rm * w)) + 4 * kA1 * kA3 * (w - rm) * (q2 + 2 * rho) + 2 * kA2 * kA3 * (2 * rm * q2 * (w**2 - 1) + rho * (3 * w * q2 + 4 * rm * (w**2 - 1)))))
    
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
