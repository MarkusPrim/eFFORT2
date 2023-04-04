import abc
import numpy as np
import numba as nb

from effort2.formfactors.kinematics import Kinematics


class FormFactorBToDstar:
    r"""This class defines the interface for any form factor parametrization to be used in conjunction with the rate implementations
for $B \to P \ell \nu_\ell$ and $B \to V \ell \nu_\ell$ decays, where P stands for Pseudoscalar and V stands for Vector heavy mesons.
    """

    def __init__(
        self, 
        m_B: float, 
        m_V: float, 
        m_L: float = 0
        ) -> None:
        r"""[summary]

        Args:
            m_B (float): Mass of the B meson.
            m_M (float): Mass of the final state meson.
            m_L (float): Mass of the final state lepton. Defaults to 0 (zero lepton mass approximation).
        """
        super().__init__()
        self.m_B = m_B
        self.m_V = m_V
        self.m_L = m_L
        self.rprime = 2 * np.sqrt(self.m_B * self.m_V) / (self.m_B + self.m_V)  # Equivalent to fDs
        self.kinematics = Kinematics(m_B, m_V, m_L)


    def A0(self, w: float) -> float:
        q2 = self.kinematics.q2(w)
        return self.R0(w) / self.rprime * self.h_A1(w)


    def A1(self, w: float) -> float:
        return (w + 1) / 2 * self.rprime * self.h_A1(w)


    def A2(self, w: float) -> float:
        return self.R2(w) / self.rprime * self.h_A1(w)


    def V(self, w: float) -> float:
        return self.R1(w) / self.rprime * self.h_A1(w)


    def Hplus(self, w: float) -> float:
        return (self.m_B + self.m_V) * self.A1(w) - 2 * self.m_B / (self.m_B + self.m_V) * self.m_V * (
                w ** 2 - 1) ** 0.5 * self.V(w)


    def Hminus(self, w: float) -> float:
        return (self.m_B + self.m_V) * self.A1(w) + 2 * self.m_B / (self.m_B + self.m_V) * self.m_V * (
                w ** 2 - 1) ** 0.5 * self.V(w)


    def Hzero(self, w: float) -> float:
        m_B = self.m_B
        m_M = self.m_V
        q2 = (m_B ** 2 + m_M ** 2 - 2 * w * m_B * m_M)
        return 1 / (2 * m_M * q2 ** 0.5) * ((m_B ** 2 - m_M ** 2 - q2) * (m_B + m_M) * self.A1(w)
                                            - 4 * m_B ** 2 * m_M ** 2 * (w ** 2 - 1) / (m_B + m_M) * self.A2(w))


    def Hscalar(self, w) -> None:
        q2 = self.kinematics.q2(w)
        m_B = self.m_B
        return 2 * m_B * self.kinematics.p(q2) / q2 ** 0.5 * self.A0(w)


    @abc.abstractmethod
    def h_A1(self, w: float) -> float:
        pass


    @abc.abstractmethod
    def R0(self, w: float) -> float:
        pass


    @abc.abstractmethod
    def R1(self, w: float) -> float:
        pass


    @abc.abstractmethod
    def R2(self, w: float) -> float:
        pass


    #@functools.lru_cache()
    @staticmethod
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


if __name__ == "__main__":
    pass
