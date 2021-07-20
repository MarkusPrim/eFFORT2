import abc

class FormFactor(abc.ABC):

    def __init__(self, m_B, m_M) -> None:
        r"""This class defines the interface to the rate implementations.

        All form factor implementations have to provide Hplus, Hminus, Hzero (, Hscalar).

        Args:
            m_B (float): Mass of the B meson.
            m_M (float): Mass of the final state meson.
        """
        super().__init__()
        self.m_B = m_B
        self.m_M = m_M
        self.w_min = 1
        self.w_max = (m_B ** 2 + m_M ** 2) / (2 * m_B * m_M)


    @abc.abstractmethod
    def Hplus(self):
        r"""Helicity ampltiude $H_+$."""
        pass

    @abc.abstractmethod
    def Hminus(self):
        r"""Helicity ampltiude $H_-$."""
        pass

    @abc.abstractmethod
    def Hzero(self):
        r"""Helicity ampltiude $H_0$."""
        pass

    @abc.abstractmethod
    def Hscalar(self):
        r"""Helicity ampltiude $H_s$. Only relevant when *not* using the zero lepton mass approximation in the rate expression."""
        pass
