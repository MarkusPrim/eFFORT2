import abc
import numpy as np

from effort2.formfactors.formFactorBase import FormFactor


class FormFactorBToC(FormFactor):
    r"""This class defines the interface for any form factor parametrization to be used in conjunction with the rate implementations
for $B \to P \ell \nu_\ell$ and $B \to V \ell \nu_\ell$ decays, where P stands for Pseudoscalar and V stands for Vector heavy mesons.
    """

    def __init__(self, m_B, m_M) -> None:
        r"""[summary]

        Args:
            m_B (float): Mass of the B meson.
            m_M (float): Mass of the final state meson.
        """
        super(FormFactorBToC, self).__init__(m_B, m_M)
        self.rprime = 2 * np.sqrt(self.m_B * self.m_M) / (self.m_B + self.m_M)


    def A0(self, w: float) -> float:
        raise RuntimeError("Not implemented. But also not required for light leptons.")


    def A1(self, w: float) -> float:
        return (w + 1) / 2 * self.rprime * self.h_A1(w)


    def A2(self, w: float) -> float:
        return self.R2(w) / self.rprime * self.h_A1(w)


    def V(self, w: float) -> float:
        return self.R1(w) / self.rprime * self.h_A1(w)


    def Hplus(self, w: float) -> float:
        return (self.m_B + self.m_M) * self.A1(w) - 2 * self.m_B / (self.m_B + self.m_M) * self.m_M * (
                w ** 2 - 1) ** 0.5 * self.V(w)


    def Hminus(self, w: float) -> float:
        return (self.m_B + self.m_M) * self.A1(w) + 2 * self.m_B / (self.m_B + self.m_M) * self.m_M * (
                w ** 2 - 1) ** 0.5 * self.V(w)


    def Hzero(self, w: float) -> float:
        m_B = self.m_B
        m_M = self.m_M
        q2 = (m_B ** 2 + m_M ** 2 - 2 * w * m_B * m_M)
        return 1 / (2 * m_M * q2 ** 0.5) * ((m_B ** 2 - m_M ** 2 - q2) * (m_B + m_M) * self.A1(w)
                                            - 4 * m_B ** 2 * m_M ** 2 * (w ** 2 - 1) / (m_B + m_M) * self.A2(w))


    def Hscalar(self) -> None:
        raise RuntimeError("Not implemented. But also not required for light leptons.")


    @abc.abstractmethod
    def h_A1(self, w: float) -> float:
        pass


    @abc.abstractmethod
    def R0(self, w: float) -> float:
        raise RuntimeError("Not implemented. But also not required for light leptons.")


    @abc.abstractmethod
    def R1(self, w: float) -> float:
        pass


    @abc.abstractmethod
    def R2(self, w: float) -> float:
        pass


    def z(self, w: float) -> float:
        """Variable for the expansion used in BGL and CLN.

        Args:
            w (float): [description]

        Returns:
            float: [description]
        """
        term1 = np.sqrt(w + 1)
        term2 = np.sqrt(2)
        return (term1 - term2) / (term1 + term2)


class BToDStarCLN(FormFactorBToC):

    def __init__(
        self,
        m_B: float,
        m_M: float,
        h_A1_1: float,
        rho2: float, 
        R1_1: float, 
        R2_1: float,
        ):
        """[summary]

        Args:
            m_B (float): [description]
            m_M (float): [description]
            h_A1_1 (float, optional): [description]
            rho2 (float, optional): [description]
            R1_1 (float, optional): [description]
            R2_1 (float, optional): [description]
        """
        super(BToDStarCLN, self).__init__(m_B, m_M)
        self.h_A1_1 = h_A1_1
        self.rho2 = rho2
        self.R1_1 = R1_1
        self.R2_1 = R2_1


    def h_A1(self, w: float) -> float:
        """[summary]

        Args:
            w (float): [description]

        Returns:
            float: [description]
        """
        rho2 = self.rho2
        z = self.z(w)
        return self.h_A1_1 * (1 - 8 * rho2 * z + (53 * rho2 - 15) * z ** 2 - (231 * rho2 - 91) * z ** 3)

    def R0(self) -> None:
        raise RuntimeError("Not implemented. But also not required for light leptons.")


    def R1(self, w: float) -> float:
        """[summary]

        Args:
            w (float): [description]

        Returns:
            float: [description]
        """
        return self.R1_1 - 0.12 * (w - 1) + 0.05 * (w - 1) ** 2


    def R2(self, w: float) -> float:
        """[summary]

        Args:
            w (float): [description]

        Returns:
            float: [description]
        """
        return self.R2_1 + 0.11 * (w - 1) - 0.06 * (w - 1) ** 2


bToDStarCLN_1702_01521v2 = BToDStarCLN(m_B=5.27963, m_M=2.01026, h_A1_1=0.906, rho2=1.03, R1_1=1.38, R2_1=0.97)
