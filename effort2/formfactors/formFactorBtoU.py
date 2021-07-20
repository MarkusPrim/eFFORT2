import abc


class FormFactorBToU(FormFactor):
    r"""This class defines the interface for any form factor parametrization to be used in conjunction with the rate implementations
for $B \to P \ell \nu_\ell$ and $B \to V \ell \nu_\ell$ decays, where P stands for Pseudoscalar and V stands for Vector light mesons.
    """

    def __init__(self, m_B, m_M) -> None:
        r"""[summary]

        Args:
            m_B (float): Mass of the B meson.
            m_M (float): Mass of the final state meson.
        """
        super(FormFactor, self).__init__(m_B, m_M)
        
        
    def kaellen(self, q2):
        return ((self.m_B + self.m_M) ** 2 - q2) * ((self.m_B - self.m_M) ** 2 - q2)


    def Hplus(self, q2: float) -> float:
        r"""Helicity ampltiude $H_+$.

        Args:
            q2 ([type]): momentum transfer to the leptonic system

        Returns:
            [type]: absolute value of the helicity amplitude
        """
        return self.kaellen(q2) ** 0.5 * self.V(q2) / (self.m_B + self.m_M) + (self.m_B + self.m_M) * self.A1(q2)


    def Hminus(self, q2: float) -> float:
        r"""Helicity ampltiude $H_-$.

        Args:
            q2 ([type]): momentum transfer to the leptonic system

        Returns:
            [type]: absolute value of the helicity amplitude
        """
        return self.kaellen(q2) ** 0.5 * self.V(q2) / (self.m_B + self.m_M) - (self.m_B + self.m_M) * self.A1(q2)


    def Hzero(self, q2: float) -> float:
        r"""Helicity ampltiude $H_0$.

        Args:
            q2 ([type]): momentum transfer to the leptonic system

        Returns:
            [type]: absolute value of the helicity amplitude
        """
        return 8 * self.m_B * self.m_M / q2 ** 0.5 * self.A12(q2)


    def Hscalar(self, q2: float) -> float:
        r"""Helicity ampltiude $H_s$. Only relevant when *not* using the zero lepton mass approximation in the rate expression.

        Args:
            q2 ([type]): momentum transfer to the leptonic system

        Returns:
            [type]: absolute value of the helicity amplitude
        """
        return self.kaellen(q2) ** 0.5 / q2 ** 0.5 * self.A0(q2)


    @abc.abstractmethod
    def A0(self, q2: float) -> float:
        pass


    @abc.abstractmethod
    def A1(self, q2: float) -> float:
        pass


    @abc.abstractmethod
    def A12(self, q2: float) -> float:
        pass


    @abc.abstractmethod
    def V(self, q2: float) -> float:
        pass
