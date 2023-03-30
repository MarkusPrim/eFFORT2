from effort2.math.integrate import quad


class AngularCoefficientsDpi:

    def __init__(
        self, 
        FF: None
        ):
        """Angular Coefficients from helicity amplitudes. Implementation of Eq. 32 in 1801.10468.

        TODO: Has some terms with the lepton mass, but implementation sets Hscalar to zero. Needs to be revisited in case of non-zero lepton masses.

        Args:
            FF (None): Form factor class which has to provide the helicity amplitudes.
        """
        self.FF = FF


    def DJ_Dw(
        self,
        J, 
        wmin: float = None,
        wmax: float = None,
        ):
        q2min = self.FF.kinematics.q2(wmax)
        q2max = self.FF.kinematics.q2(wmin)
        return quad(J, q2min, q2max)[0]


    def F(self, q2):
        return (2 / self.FF.m_B ** 3) * (3 * self.FF.kinematics.p(q2)) / (128 * 2 ** 4 * self.FF.m_B ** 2)


    def J_1s(self, q2):
        w = self.FF.kinematics.w(q2)
        Hplus = self.FF.Hplus(w)
        Hminus = self.FF.Hminus(w)
        return self.F(q2) * (0.5 * (Hplus ** 2 + Hminus ** 2) * (self.FF.m_L ** 2 + 3 * q2))


    def J_1c(self, q2):
        w = self.FF.kinematics.w(q2)
        Hzero = self.FF.Hzero(w)
        Hscalar = 0
        return self.F(q2) * (2 * (2 * self.FF.m_L ** 2 * Hscalar ** 2 + Hzero ** 2 * (self.FF.m_L ** 2 + q2)))


    def J_2s(self, q2):
        w = self.FF.kinematics.w(q2)
        Hplus = self.FF.Hplus(w)
        Hminus = self.FF.Hminus(w)
        return self.F(q2) * (0.5 * (Hplus ** 2 + Hminus ** 2) * (q2 - self.FF.m_L ** 2))


    def J_2c(self, q2):
        w = self.FF.kinematics.w(q2)
        Hzero = self.FF.Hzero(w)
        return self.F(q2) * (2 * Hzero ** 2 * (self.FF.m_L ** 2 - q2))


    def J_3(self, q2):
        w = self.FF.kinematics.w(q2)
        Hplus = self.FF.Hplus(w)
        Hminus = self.FF.Hminus(w)
        return self.F(q2) * (2 * Hplus * Hminus * (self.FF.m_L ** 2 - q2))


    def J_4(self, q2):
        w = self.FF.kinematics.w(q2)
        Hplus = self.FF.Hplus(w)
        Hminus = self.FF.Hminus(w)
        Hzero = self.FF.Hzero(w)
        return -self.F(q2) * (Hzero * (Hplus + Hminus) * (self.FF.m_L ** 2 - q2))


    def J_5(self, q2):
        w = self.FF.kinematics.w(q2)
        Hplus = self.FF.Hplus(w)
        Hminus = self.FF.Hminus(w)
        Hzero = self.FF.Hzero(w)
        Hscalar = 0
        return self.F(q2) * (-2 * (Hplus + Hminus) * Hscalar * self.FF.m_L ** 2 - 2 * Hzero * (Hplus - Hminus) * q2)


    def J_6s(self, q2):
        w = self.FF.kinematics.w(q2)
        Hplus = self.FF.Hplus(w)
        Hminus = self.FF.Hminus(w)
        return -self.F(q2) * (2 * (Hplus ** 2 - Hminus ** 2) * q2)


    def J_6c(self, q2):
        w = self.FF.kinematics.w(q2)
        Hplus = self.FF.Hplus(w)
        Hminus = self.FF.Hminus(w)
        Hzero = self.FF.Hzero(w)
        Hscalar = 0
        return -self.F(q2) * (-8 * Hzero * Hscalar * self.FF.m_L ** 2)


    def J_7(self, q2):
        return self.F(q2) * (0)


    def J_8(self, q2):
        raise NotImplemented


    def J_9(self, q2):
        raise NotImplemented


class AngularCoefficientsDgamma:

    def __init__(
        self, 
        FF: None
        ):
        """Angular Coefficients from helicity amplitudes. Implementation of Eq. 33 in 1801.10468.

        TODO: Has some terms with the lepton mass, but implementation sets Hscalar to zero. Needs to be revisited in case of non-zero lepton masses.

        Args:
            FF (None): Form factor class which has to provide the helicity amplitudes.
        """
        self.FF = FF
