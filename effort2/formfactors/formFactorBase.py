import abc

class FormFactor(abc.ABC):

    def __init__(self, m_B, m_M, m_L) -> None:
        r"""This class defines the interface to the rate implementations.

        All form factor implementations have to provide Hplus, Hminus, Hzero (, Hscalar).

        Args:
            m_B (float): Mass of the B meson.
            m_M (float): Mass of the final state meson.
            m_L (float): Mass of the final state lepton.
        """
        super().__init__()
        self.m_B = m_B
        self.m_M = m_M
        self.m_L = m_L
        self.w_min = 1
        self.w_max = (m_B ** 2 + m_M ** 2) / (2 * m_B * m_M)


    @abc.abstractmethod
    def Hplus(self, w):
        r"""Helicity ampltiude $H_+$."""
        pass

    @abc.abstractmethod
    def Hminus(self, w):
        r"""Helicity ampltiude $H_-$."""
        pass

    @abc.abstractmethod
    def Hzero(self, w):
        r"""Helicity ampltiude $H_0$."""
        pass

    @abc.abstractmethod
    def Hscalar(self, w):
        r"""Helicity ampltiude $H_s$. Only relevant when *not* using the zero lepton mass approximation in the rate expression."""
        pass

    
    def q2(self, w):
        return self.m_B ** 2 + self.m_M ** 2 - 2 * w * self.m_B * self.m_M

    def w(self, q2):
        return (self.m_B ** 2 + self.m_M ** 2 - q2) / (2 * self.m_B * self.m_M)
