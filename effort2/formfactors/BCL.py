class BToPLNuBCL(BToPLNu):

    def __init__(self, m_B: float, m_P: float, m_L: float, V_ub: float, eta_EW: float = 1.0066):
        super(BToPLNuBCL, self).__init__(m_B, m_P, m_L, V_ub, eta_EW)
        self._coefficients = None
        self.mBstar = 5.325

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, coefficients):
        self._coefficients = coefficients

    def fzero(self, q2):
        N = 4
        return sum([b * self.z(q2) ** n for n, b in enumerate(self._coefficients[N:])])

    def fplus(self, q2):
        N = 4
        return 1 / (1 - q2 / self.mBstar ** 2) * sum(
            [b * (self.z(q2) ** n - (-1) ** (n - N) * n / N * self.z(q2) ** N) for n, b in
             enumerate(self._coefficients[:N])]
        )

    def Hzero(self, q2):  # Could be the implementation of parent 
        return 2 * self.m_B * self.pion_momentum(q2) / np.sqrt(q2) * self.fplus(q2)

    def Hscalar(self, q2):  # Could be the implementation of parent
        return (self.m_B ** 2 - self.m_P ** 2) / np.sqrt(q2) * self.fzero(q2)
