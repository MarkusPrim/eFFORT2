import abc
import numpy as np

from effort2.formfactors.kinematics import Kinematics


class HelicityBasisBToP:
    """Helicity basis for B to Pseudoscalar transitions."""

    def __init__(
        self, 
        m_B: float, 
        m_P: float, 
        m_L: float = 0
        ) -> None:
        r"""[summary]

        Args:
            m_B (float): Mass of the B meson.
            m_P (float): Mass of the final state meson.
            m_L (float): Mass of the final state lepton. Defaults to 0 (zero lepton mass approximation).
        """
        super().__init__()
        self.m_B = m_B
        self.m_P = m_P
        self.m_L = m_L
        self.kinematics = Kinematics(m_B, m_P, m_L)


    def Hzero(self, w: float) -> float:
        q2 = self.kinematics.q2(w)
        return (self.m_B * self.m_P) ** 0.5 * (self.m_B + self.m_P) / q2 ** 0.5 * (w**2-1) ** 0.5 * self.V(w)


    def Hscalar(self, w: float) -> float:
        q2 = self.kinematics.q2(w)
        return (self.m_B*self.m_P) ** 0.5 * (self.m_B - self.m_P) / q2 ** 0.5 * (w + 1) * self.S(w)
    

    @abc.abstractmethod
    def V(self, w: float) -> float:
        return 0
    
    
    @abc.abstractmethod
    def S(self, w: float) -> float:
        return 0


class HelicityBasisBToV:
    """Helicity basis for B to Vector transitions."""

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
        self.kinematics = Kinematics(m_B, m_V, m_L)


    def Hplus(self, w: float) -> float:
        return (self.m_B + self.m_V) * self.A1(w) - 2 * self.m_B / (self.m_B + self.m_V) * self.m_V * (
                w ** 2 - 1) ** 0.5 * self.V(w)


    def Hminus(self, w: float) -> float:
        return (self.m_B + self.m_V) * self.A1(w) + 2 * self.m_B / (self.m_B + self.m_V) * self.m_V * (
                w ** 2 - 1) ** 0.5 * self.V(w)


    def Hzero(self, w: float) -> float:
        m_B = self.m_B
        m_M = self.m_V
        q2 = self.kinematics.q2(w)
        return 1 / (2 * m_M * q2 ** 0.5) * ((m_B ** 2 - m_M ** 2 - q2) * (m_B + m_M) * self.A1(w)
                                            - 4 * m_B ** 2 * m_M ** 2 * (w ** 2 - 1) / (m_B + m_M) * self.A2(w))


    def Hscalar(self, w) -> None:
        q2 = self.kinematics.q2(w)
        m_B = self.m_B
        return 2 * m_B * self.kinematics.p(q2) / q2 ** 0.5 * self.A0(w)


    @abc.abstractmethod
    def A0(self, w: float) -> float:
        return 0

    
    @abc.abstractmethod
    def A1(self, w: float) -> float:
        return 0


    @abc.abstractmethod
    def A2(self, w: float) -> float:
        return 0


    @abc.abstractmethod
    def V(self, w: float) -> float:
        return 0


class FormFactorHQETBToP(HelicityBasisBToP):
    """Defines the form factors V and S for the B to Pseudoscalar transitions. 
    
    Implements the translation from the HQET basis f0 and f+ to V and S.
    
    Nota bene: We call this HQET basis for simplicity of naming. The actualy HQET basis is h+ and h-,
    however, it is more convenient to use f0 and f+ which are a linear combination of h+ and h-.
    The advantage is, that the mass independent and dependent terms rely on f0 and f+ _only_ respectivly,
    where in the h+ and h- basis these two terms do not decouple.
    """

    def __init__(
        self, 
        m_B: float, 
        m_P: float, 
        m_L: float = 0
        ) -> None:
        r"""[summary]

        Args:
            m_B (float): Mass of the B meson.
            m_P (float): Mass of the final state meson.
            m_L (float): Mass of the final state lepton. Defaults to 0 (zero lepton mass approximation).
        """
        super().__init__(m_B, m_P, m_L)
        self.r = self.m_P / self.m_B
    

    def V(self, w: float) -> float:
        return self.hplus(w) - (self.m_B - self.m_P) / (self.m_B + self.m_P) * self.hminus(w)


    def S(self, w: float) -> float:
        return self.hplus(w) - (self.m_B + self.m_P) / (self.m_B - self.m_P) * (w - 1) / (w + 1) * self.hminus(w)


    def hplus(self, w: float) -> float:
        r = self.r
        f0 = self.fzero(w)
        fp = self.fplus(w)
        return ((1 + r) * (f0 - 2 * f0 * r + 2 * fp * r + f0 * r ** 2 - 2 * fp * r * w)) / (2 * r ** 0.5 * (1 + r ** 2 - 2 * r * w))


    def hminus(self, w: float) -> float:
        r = self.r
        f0 = self.fzero(w)
        fp = self.fplus(w)
        return ((-1 + r) * (-f0 - 2 * f0 * r + 2 * fp * r - f0 * r ** 2 + 2 * fp * r * w)) / (2 * r ** 0.5 * (1 + r ** 2 - 2 * r * w))


    @abc.abstractmethod
    def fzero(self, w: float) -> float:
        pass

    
    @abc.abstractmethod
    def fplus(self, w:float) -> float:
        pass      


class FormFactorHQETBToV(HelicityBasisBToV):
    """Defines the form factors A0, A1, A2, and V for the B to Vector transitions. 
    
    Implements the translation from the HQET basis h_A1, R0, R1, R2 to A0, A1, A2, and V.
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
            m_V (float): Mass of the final state meson.
            m_L (float): Mass of the final state lepton. Defaults to 0 (zero lepton mass approximation).
        """
        super().__init__(m_B, m_V, m_L)
        self.rprime = 2 * np.sqrt(self.m_B * self.m_V) / (self.m_B + self.m_V)  # Equivalent to fDs
    

    def A0(self, w: float) -> float:
        return self.R0(w) / self.rprime * self.h_A1(w)


    def A1(self, w: float) -> float:
        return (w + 1) / 2 * self.rprime * self.h_A1(w)


    def A2(self, w: float) -> float:
        return self.R2(w) / self.rprime * self.h_A1(w)


    def V(self, w: float) -> float:
        return self.R1(w) / self.rprime * self.h_A1(w)


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


if __name__ == "__main__":
    pass
