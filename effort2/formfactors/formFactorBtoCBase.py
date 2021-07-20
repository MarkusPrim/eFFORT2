import abc


class FormFactorBToC(abc.ABC):
    r"""This class defines the interface for any form factor parametrization to be used in conjunction with the rate implementations
for $B \to P \ell \nu_\ell$ and $B \to V \ell \nu_\ell$ decays, where P stands for Pseudoscalar and V stands for Vector heavy mesons.
    """

    def __init__(self, m_B, m_M) -> None:
        r"""[summary]

        Args:
            m_B (float): Mass of the B meson.
            m_M (float): Mass of the final state meson.
        """
        super(FormFactor, self).__init__(m_B, m_M)


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
