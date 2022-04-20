import numpy as np


class Kinematics:

    def __init__(self, m_B, m_M, m_L, numerical_epsilon=1e-8) -> None:
        r"""This class defines common kinematic variables.

        Args:
            m_B (float): Mass of the B meson.
            m_M (float): Mass of the final state meson.
            m_L (float): Mass of the final state lepton.
            numerical_epsilon (float): Epsilon value used to avoid numerical stability in some applications.
        """
        super().__init__()
        self.m_B = m_B
        self.m_M = m_M
        self.m_L = m_L
        self.numerical_epsilon = numerical_epsilon

        self.w_min = 1
        self.w_max = (m_B ** 2 + m_M ** 2- m_L**2) / (2 * m_B * m_M)
        self.w_range = (self.w_min, self.w_max)
        self.w_range_numerical_stable = (self.w_min + numerical_epsilon, self.w_max - numerical_epsilon)
        
        self.cosL_min = -1
        self.cosL_max = 1
        self.cosL_range = self.cosL_min, self.cosL_max

        self.cosV_min = -1
        self.cosV_max = 1
        self.cosV_range = self.cosV_min, self.cosV_max

        self.chi_min = 0
        self.chi_max = 2*np.pi
        self.chi_range = self.chi_min, self.chi_max        

    
    def p(self, q2):
        return np.sqrt( ( (self.m_B**2 + self.m_M**2 - q2)/(2*self.m_B) )**2 - self.m_M**2 )

    def q2(self, w):
        return self.m_B ** 2 + self.m_M ** 2 - 2 * w * self.m_B * self.m_M

    def w(self, q2):
        return (self.m_B ** 2 + self.m_M ** 2 - q2) / (2 * self.m_B * self.m_M)


if __name__ == "__main__":
    pass