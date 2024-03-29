from effort2.math.integrate import quad

import numpy as np


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


    def Norm(self):
        wmin = self.FF.kinematics.w_min
        wmax = self.FF.kinematics.w_max
        return 8 / 9 * np.pi * (
            3 * self.DJ_Dw(self.J1c, wmin, wmax) 
            + 6 * self.DJ_Dw(self.J1s, wmin, wmax) 
            - self.DJ_Dw(self.J2c, wmin, wmax) 
            - 2 * self.DJ_Dw(self.J2s, wmin, wmax)
            )


    def F(self, q2):
        return (2 / self.FF.m_B ** 3) * (3 * self.FF.kinematics.p(q2)) / (128 * 2 ** 4 * self.FF.m_B ** 2)


    def J1s(self, q2):
        w = self.FF.kinematics.w(q2)
        Hplus = self.FF.Hplus(w)
        Hminus = self.FF.Hminus(w)
        return self.F(q2) * (0.5 * (Hplus ** 2 + Hminus ** 2) * (self.FF.m_L ** 2 + 3 * q2))


    def J1c(self, q2):
        w = self.FF.kinematics.w(q2)
        Hzero = self.FF.Hzero(w)
        Hscalar = self.FF.Hscalar(w)
        return self.F(q2) * (2 * (2 * self.FF.m_L ** 2 * Hscalar ** 2 + Hzero ** 2 * (self.FF.m_L ** 2 + q2)))


    def J2s(self, q2):
        w = self.FF.kinematics.w(q2)
        Hplus = self.FF.Hplus(w)
        Hminus = self.FF.Hminus(w)
        return self.F(q2) * (0.5 * (Hplus ** 2 + Hminus ** 2) * (q2 - self.FF.m_L ** 2))


    def J2c(self, q2):
        w = self.FF.kinematics.w(q2)
        Hzero = self.FF.Hzero(w)
        return self.F(q2) * (2 * Hzero ** 2 * (self.FF.m_L ** 2 - q2))


    def J3(self, q2):
        w = self.FF.kinematics.w(q2)
        Hplus = self.FF.Hplus(w)
        Hminus = self.FF.Hminus(w)
        return self.F(q2) * (2 * Hplus * Hminus * (self.FF.m_L ** 2 - q2))


    def J4(self, q2):
        w = self.FF.kinematics.w(q2)
        Hplus = self.FF.Hplus(w)
        Hminus = self.FF.Hminus(w)
        Hzero = self.FF.Hzero(w)
        return -self.F(q2) * (Hzero * (Hplus + Hminus) * (self.FF.m_L ** 2 - q2))


    def J5(self, q2):
        w = self.FF.kinematics.w(q2)
        Hplus = self.FF.Hplus(w)
        Hminus = self.FF.Hminus(w)
        Hzero = self.FF.Hzero(w)
        Hscalar = self.FF.Hscalar(w)
        return self.F(q2) * (-2 * (Hplus + Hminus) * Hscalar * self.FF.m_L ** 2 - 2 * Hzero * (Hplus - Hminus) * q2)


    def J6s(self, q2):
        w = self.FF.kinematics.w(q2)
        Hplus = self.FF.Hplus(w)
        Hminus = self.FF.Hminus(w)
        return -self.F(q2) * (2 * (Hplus ** 2 - Hminus ** 2) * q2)


    def J6c(self, q2):
        w = self.FF.kinematics.w(q2)
        Hzero = self.FF.Hzero(w)
        Hscalar = self.FF.Hscalar(w)
        return -self.F(q2) * (-8 * Hzero * Hscalar * self.FF.m_L ** 2)


    def J7(self, q2):
        return 0
        raise NotImplemented


    def J8(self, q2):
        return 0
        raise NotImplemented


    def J9(self, q2):
        return 0
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


if __name__ == "__main__":
    pass
