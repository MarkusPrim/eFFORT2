from ast import Lambda
import numpy as np
from effort2.formfactors.DStStAlphaSCorrections import DStStAlphaSCorrections


class BToDStarStarBroad(DStStAlphaSCorrections):
    """
    A class to compute the form factors for the two broad D** states (D0* and D1')
    """

    def __init__(
        self,
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
        super().__init__(m_c, m_b)
        self.m_c = m_c
        self.m_b = m_b
        self.alphaS = alphaS / np.pi
        self.LambdaBar = LambdaBar
        self.LambdaBarPrime = LambdaBarPrime
        self.LambdaBarStar = LambdaBarStar
        self.chi1 = chi1
        self.chi2 = chi2

        self.Z1, self.zetap, self.zeta1 = self.set_model_parameters(params)
        # Scale mu = sqrt(mc*mb)
        # Certain values depend on renorm scheme
        # The chromomagnetic terms chi1/2 are neglected in Hammer and certain Approximations.
        # I'm not sure whether they should be set to 0 with the most up-to-date fit of the 3 zeta parameters.

    def set_model_parameters(
        self,
        params: tuple,
    ):
        """
        Sets the input model parameters zeta(1), zeta' and zeta1 (cf. Table V in arxiv:1711.03110).
        Expects the parameters in that order.

        Args:
            params ([tuple]): input parameters for the broad D** form factor.
        """
        Z1, zetap, zeta1 = params
        return Z1, zetap, zeta1

    def Gb(
        self,
        w: float,
    ):
        """
        Used in all B -> D** broad form factors
        """
        LambdaBar = self.LambdaBar
        LambdaBarStar = self.LambdaBarStar
        zeta1 = self.zeta1
        return ((1 + 2 * w) * LambdaBarStar - (2 + w) * LambdaBar) / (w + 1) - 2 * (w - 1) * zeta1

    def gP(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        LambdaBar = self.LambdaBar
        LambdaBarStar = self.LambdaBarStar
        chi1 = self.chi1
        chi2 = self.chi2
        epsilonC = 1 / (2 * self.m_c)
        epsilonB = 1 / (2 * self.m_b)
        zeta1 = self.zeta1
        CP = self.CP(w)
        Gb = self.Gb(w)
        return (w - 1) * (1 + alphaS * CP) + epsilonC * (3 * (w * LambdaBarStar - LambdaBar) - 2 * (w**2 - 1) * zeta1 + (w -1) * (6 * chi1 - 2 * (w + 1) * chi2)) - epsilonB * (w + 1) * Gb

    def gplus(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        LambdaBar = self.LambdaBar
        LambdaBarStar = self.LambdaBarStar
        epsilonC = 1 / (2 * self.m_c)
        epsilonB = 1 / (2 * self.m_b)
        zeta1 = self.zeta1
        CA2 = self.CA2(w)
        CA3 = self.CA3(w)
        Gb = self.Gb(w)
        return 0.5 * alphaS * (w - 1) * (CA2 + CA3) - epsilonC * (3 * (w * LambdaBarStar - LambdaBar) / (w + 1) - 2 * (w - 1) * zeta1) - epsilonB * Gb

    def gminus(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        epsilonC = 1 / (2 * self.m_c)
        chi1 = self.chi1
        chi2 = self.chi2
        CA1 = self.CA1(w)
        CA2 = self.CA2(w)
        CA3 = self.CA3(w)
        return 1 + alphaS * (CA1 + 0.5 * (w - 1) * (CA2 - CA3)) + epsilonC * (6 * chi1 - 2 * (w + 1) * chi2)

    def gT(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        LambdaBar = self.LambdaBar
        LambdaBarStar = self.LambdaBarStar
        chi1 = self.chi1
        chi2 = self.chi2
        epsilonC = 1 / (2 * self.m_c)
        epsilonB = 1 / (2 * self.m_b)
        zeta1 = self.zeta1
        Gb = self.Gb(w)
        CT1 = self.CT1(w)
        return 1 + alphaS * CT1 + epsilonC * (3 * (w * LambdaBarStar - LambdaBar) / (w + 1) - 2 * (w - 1) * zeta1 + 6 * chi1 - 2 * (w + 1) * chi2) - epsilonB * Gb

    def gS(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        LambdaBar = self.LambdaBar
        LambdaBarStar = self.LambdaBarStar
        chi1 = self.chi1
        chi2 = self.chi2
        epsilonC = 1 / (2 * self.m_c)
        epsilonB = 1 / (2 * self.m_b)
        zeta1 = self.zeta1
        Gb = self.Gb(w)
        CS = self.CS(w)
        return 1 + alphaS * CS - epsilonC * ((w * LambdaBarStar - LambdaBar) / (w + 1) - 2 * (w - 1) * zeta1 + 2 * chi1 - 2 * (w + 1) * chi2) - epsilonB * Gb
    
    def gV1(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        LambdaBar = self.LambdaBar
        LambdaBarStar = self.LambdaBarStar
        chi1 = self.chi1
        epsilonC = 1 / (2 * self.m_c)
        epsilonB = 1 / (2 * self.m_b)
        Gb = self.Gb(w)
        CV1 = self.CV1(w)
        return (w - 1) * (1 + alphaS * CV1) + epsilonC * (w * LambdaBarStar - LambdaBar - 2 * (w - 1) * chi1) - epsilonB * (w + 1) * Gb
    
    def gV2(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        chi2 = self.chi2
        epsilonC = 1 / (2 * self.m_c)
        zeta1 = self.zeta1
        CV2 = self.CV2(w)
        return - alphaS * CV2 + 2 * epsilonC * (zeta1 - chi2)

    def gV3(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        LambdaBar = self.LambdaBar
        LambdaBarStar = self.LambdaBarStar
        chi1 = self.chi1
        chi2 = self.chi2
        epsilonC = 1 / (2 * self.m_c)
        epsilonB = 1 / (2 * self.m_b)
        zeta1 = self.zeta1
        Gb = self.Gb(w)
        CV1 = self.CV1(w)
        CV3 = self.CV3(w)
        return -1 - alphaS * (CV1 + CV3) - epsilonC * ((w * LambdaBarStar - LambdaBar) / (w + 1) + 2 * (zeta1 - chi1 + chi2)) + epsilonB * Gb
    
    def gA(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        LambdaBar = self.LambdaBar
        LambdaBarStar = self.LambdaBarStar
        chi1 = self.chi1
        epsilonC = 1 / (2 * self.m_c)
        epsilonB = 1 / (2 * self.m_b)
        Gb = self.Gb(w)
        CA1 = self.CA1(w)
        return 1 + alphaS * CA1 + epsilonC * ((w * LambdaBarStar - LambdaBar) / (w + 1) - 2 * chi1) - epsilonB * Gb
    
    def gT1(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        LambdaBar = self.LambdaBar
        LambdaBarStar = self.LambdaBarStar
        chi1 = self.chi1
        epsilonC = 1 / (2 * self.m_c)
        epsilonB = 1 / (2 * self.m_b)
        Gb = self.Gb(w)
        CT1 = self.CT1(w)
        CT2 = self.CT2(w)
        return -1 - alphaS * (CT1 + (w - 1) * CT2) + epsilonC * ((w * LambdaBarStar - LambdaBar) / (w + 1) + 2 * chi1) + epsilonB * Gb
    
    def gT2(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        LambdaBar = self.LambdaBar
        LambdaBarStar = self.LambdaBarStar
        chi1 = self.chi1
        epsilonC = 1 / (2 * self.m_c)
        epsilonB = 1 / (2 * self.m_b)
        Gb = self.Gb(w)
        CT1 = self.CT1(w)
        CT3 = self.CT3(w)
        return 1 + alphaS * (CT1 - (w - 1) * CT3) + epsilonC * ((w * LambdaBarStar - LambdaBar) / (w + 1) - 2 * chi1) + epsilonB * Gb

    def gT3(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        zeta1 = self.zeta1
        chi2 = self.chi2
        epsilonC = 1 / (2 * self.m_c)
        CT2 = self.CT2(w)
        return -alphaS * CT2 + 2 * epsilonC * (zeta1 + chi2)


class BToDStarStarNarrow(DStStAlphaSCorrections):
    """
    A class to compute the form factors for the two narrow D** states (D1 and D2)
    """

    def __init__(
        self,
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
    ):
        super().__init__(m_c, m_b)
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

    def set_model_parameters(
        self,
        params: tuple,
    ):
        """
        Sets the input model parameters tau(1), tau', tau1 and tau2 (cf. Table V in arxiv:1711.03110).
        Expects the parameters in that order.

        Args:
            params ([tuple]): input parameters for the narrow D** form factor.
        """
        T1, taup, tau1, tau2 = params
        return T1, taup, tau1, tau2

    def Fb(
        self,
        w: float,
    ):
        LambdaBar = self.LambdaBar
        LambdaBarStar = self.LambdaBarStar
        tau1 = self.tau1
        tau2 = self.tau2
        return LambdaBar + LambdaBarStar - (2 * w + 1) * tau1 - tau2

    # Form factors for D1
    # The f form factors have to be multiplied by sqrt(6)
    def fS(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        LambdaBar = self.LambdaBar
        LambdaBarPrime = self.LambdaBarPrime
        tau1 = self.tau1
        tau2 = self.tau2
        epsilonC = 1 / (2 * self.m_c)
        epsilonB = 1 / (2 * self.m_b)
        eta1 = self.eta1
        eta2 = self.eta2
        eta3 = self.eta3
        Fb = self.Fb(w)
        CS = self.CS(w)
        return -2 * (w + 1) * (1 + alphaS * CS) - 2 * epsilonB * (w - 1) * Fb - epsilonC * (4 * (w * LambdaBarPrime - LambdaBar) - 2 * (w - 1) * ((2 * w + 1) * tau1 + tau2) + 2 * (w + 1) * (6 * eta1 + 2 * (w - 1) * eta2 - eta3))

    def fV1(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        LambdaBar = self.LambdaBar
        LambdaBarPrime = self.LambdaBarPrime
        tau1 = self.tau1
        tau2 = self.tau2
        epsilonC = 1 / (2 * self.m_c)
        epsilonB = 1 / (2 * self.m_b)
        eta1 = self.eta1
        eta3 = self.eta3
        Fb = self.Fb(w)
        CV1 = self.CV1(w)
        return (1 - w**2) * (1 + alphaS * CV1) - epsilonB * (w**2 - 1) * Fb - epsilonC * (4 * (w + 1) * (w * LambdaBarPrime - LambdaBar) - (w**2 - 1) * (3 * tau1 - 3 * tau2 + 2 * eta1 + 3 * eta3))

    def fV2(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        tau1 = self.tau1
        tau2 = self.tau2
        epsilonC = 1 / (2 * self.m_c)
        epsilonB = 1 / (2 * self.m_b)
        eta1 = self.eta1
        eta2 = self.eta2
        eta3 = self.eta3
        Fb = self.Fb(w)
        CV1 = self.CV1(w)
        CV2 = self.CV2(w)
        return -3 - alphaS * (3 * CV1 + 2 * (w + 1) * CV2) - 3 * epsilonB * Fb - epsilonC * ((4 * w - 1) * tau1 + 5 * tau2 + 10 * eta1 + 4 * (w - 1) * eta2 - 5 * eta3)

    def fV3(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        LambdaBar = self.LambdaBar
        LambdaBarPrime = self.LambdaBarPrime
        tau1 = self.tau1
        tau2 = self.tau2
        epsilonC = 1 / (2 * self.m_c)
        epsilonB = 1 / (2 * self.m_b)
        eta1 = self.eta1
        eta2 = self.eta2
        eta3 = self.eta3
        Fb = self.Fb(w)
        CV1 = self.CV1(w)
        CV3 = self.CV3(w)
        return w - 2 - alphaS * ((2 - w) * CV1 + 2 * (w + 1) * CV3) + epsilonB * (w + 2) * Fb + epsilonC * (4 * (w * LambdaBarPrime - LambdaBar) + (w + 2)* tau1 + (2 + 3 * w) * tau2 - 2 * (w + 6) * eta1 - 4 * (w - 1) * eta2 - (3 * w - 2) * eta3)

    def fA(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        LambdaBar = self.LambdaBar
        LambdaBarPrime = self.LambdaBarPrime
        tau1 = self.tau1
        tau2 = self.tau2
        epsilonC = 1 / (2 * self.m_c)
        epsilonB = 1 / (2 * self.m_b)
        eta1 = self.eta1
        eta3 = self.eta3
        Fb = self.Fb(w)
        CA1 = self.CA1(w)
        return -(w + 1) * (1 + alphaS * CA1) - epsilonB * (w - 1) * Fb - epsilonC * (4 * (w * LambdaBarPrime - LambdaBar) - 3 * (w - 1) * (tau1 - tau2) -  (w + 1) * (2 * eta1 + 3 * eta3))

    def fT1(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        LambdaBar = self.LambdaBar
        LambdaBarPrime = self.LambdaBarPrime
        tau1 = self.tau1
        tau2 = self.tau2
        epsilonC = 1 / (2 * self.m_c)
        epsilonB = 1 / (2 * self.m_b)
        eta1 = self.eta1
        eta3 = self.eta3
        Fb = self.Fb(w)
        CT1 = self.CT1(w)
        CT2 = self.CT2(w)
        return (w + 1) * (1 + alphaS * (CT1 + (w - 1) * CT2)) + epsilonB * (w - 1) * Fb - epsilonC * (4 * (w * LambdaBarPrime - LambdaBar) - 3 * (w - 1) * (tau1 - tau2) + (w + 1) * (2 * eta1 + 3 * eta3))

    def fT2(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        LambdaBar = self.LambdaBar
        LambdaBarPrime = self.LambdaBarPrime
        tau1 = self.tau1
        tau2 = self.tau2
        epsilonC = 1 / (2 * self.m_c)
        epsilonB = 1 / (2 * self.m_b)
        eta1 = self.eta1
        eta2 = self.eta2
        eta3 = self.eta3
        Fb = self.Fb(w)
        CT1 = self.CT1(w)
        CT3 = self.CT3(w)
        return -(w + 1) * (1 + alphaS * (CT1 - (w - 1) * CT3)) + epsilonB * (w - 1) * Fb - epsilonC * (4 * (w * LambdaBarPrime - LambdaBar) - 3 * (w - 1) * (tau1 - tau2) - (w + 1) * (2 * eta1 + 3 * eta3))

    def fT3(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        tau1 = self.tau1
        tau2 = self.tau2
        epsilonC = 1 / (2 * self.m_c)
        epsilonB = 1 / (2 * self.m_b)
        eta1 = self.eta1
        eta2 = self.eta2
        eta3 = self.eta3
        Fb = self.Fb(w)
        CT1 = self.CT1(w)
        CT2 = self.CT2(w)
        CT3 = self.CT3(w)
        return 3 + alphaS * (3 * CT1 - (2 - w) * CT2 + 3 * CT3) + 3 * epsilonB * Fb - epsilonC * ((4 * w - 1) * tau1 + 5 * tau2 - 10 * eta1 - 4 * (w - 1) * eta2 + 5 * eta3)


    # Form factors for D2*
    def kP(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        tau1 = self.tau1
        tau2 = self.tau2
        epsilonC = 1 / (2 * self.m_c)
        epsilonB = 1 / (2 * self.m_b)
        eta1 = self.eta1
        eta2 = self.eta2
        eta3 = self.eta3
        Fb = self.Fb(w)
        CP = self.CP(w)
        return 1 + alphaS * CP + epsilonB * Fb + epsilonC * ((2 * w + 1) * tau1 + tau2 - 2 * eta1 - 2 * (w - 1) * eta2 + eta3)

    def kV(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        tau1 = self.tau1
        tau2 = self.tau2
        epsilonC = 1 / (2 * self.m_c)
        epsilonB = 1 / (2 * self.m_b)
        eta1 = self.eta1
        eta3 = self.eta3
        Fb = self.Fb(w)
        CV1 = self.CV1(w)
        return -1 - alphaS * CV1 - epsilonB * Fb - epsilonC * (tau1 - tau2 - 2 * eta1 + eta3)

    def kA1(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        tau1 = self.tau1
        tau2 = self.tau2
        epsilonC = 1 / (2 * self.m_c)
        epsilonB = 1 / (2 * self.m_b)
        eta1 = self.eta1
        eta3 = self.eta3
        Fb = self.Fb(w)
        CA1 = self.CA1(w)
        return -(w + 1) * (1 + alphaS * CA1) - epsilonB * (w - 1) * Fb - epsilonC * ((w - 1) * (tau1 - tau2) - (w + 1) * (2 * eta1 - eta3))

    def kA2(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        tau1 = self.tau1
        epsilonC = 1 / (2 * self.m_c)
        eta2 = self.eta2
        CA2 = self.CA2(w)
        return alphaS * CA2 - 2 * epsilonC * (tau1 + eta2)

    def kA3(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        tau1 = self.tau1
        tau2 = self.tau2
        epsilonC = 1 / (2 * self.m_c)
        epsilonB = 1 / (2 * self.m_b)
        eta1 = self.eta1
        eta2 = self.eta2
        eta3 = self.eta3
        Fb = self.Fb(w)
        CA1 = self.CA1(w)
        CA3 = self.CA3(w)
        return 1 + alphaS * (CA1 + CA3) + epsilonB * Fb - epsilonC * (tau1 + tau2 + 2 * eta1 - 2 * eta2 - eta3)

    def kT1(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        epsilonC = 1 / (2 * self.m_c)
        eta1 = self.eta1
        eta3 = self.eta3
        CT1 = self.CT1(w)
        CT2 = self.CT2(w)
        CT3 = self.CT3(w)
        return 1 + alphaS * (CT1 + 0.5 * (w - 1) * (CT2 - CT3)) - epsilonC * (2 * eta1 - eta3)

    def kT2(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        tau1 = self.tau1
        tau2 = self.tau2
        epsilonC = 1 / (2 * self.m_c)
        epsilonB = 1 / (2 * self.m_b)
        Fb = self.Fb(w)
        CT2 = self.CT2(w)
        CT3 = self.CT3(w)
        return 0.5 * alphaS * (w + 1) * (CT2 + CT3) + epsilonB * Fb - epsilonC * (tau1 - tau2)

    def kT3(
        self,
        w: float,
    ):
        alphaS = self.alphaS
        tau1 = self.tau1
        epsilonC = 1 / (2 * self.m_c)
        eta2 = self.eta2
        CT2 = self.CT2(w)
        return -alphaS * CT2 + 2 * epsilonC * (tau1 - eta2)


if __name__ == "__main__":
    pass
