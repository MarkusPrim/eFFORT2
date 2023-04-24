import numpy as np
from effort2.formfactors.kinematics import Kinematics
from effort2.math.integrate import quad


class BtoP:

    def __init__(
        self, 
        FF: None,
        Vxb: float, 
        m_B: float = None, 
        m_P: float = None,
        m_L: float = None,
        G_F: float = 1.1663787e-5,
        eta_EW: float = 1.0066,
        BR_D_decay: float = 1,
        ):
        """Initialize a class for calculating decay rate of B to Vector meson decays.

        TODO: 
            * Add Glebsch Gordan coefficient to differ between B+ and B0 decays.

        Nota bene:
            * If you use a form factor parametrization where you can/want to absorb eta_EW and/or Vcb into the definition of the form factor, initialize this class setting both values as 1.
            * The class does not provide marginalizations over w, because this would make the analytical integration dependent on the chosen form factor parametrization. Numerical integration over w is therefore required in many applications.
            * Using the exact boundaries of w on the integration process might cause issues. Try adding/subtracting an epsilon to wmin/wmax to resolve the issue.

        Args:
            Vxb (float): CKM parameter Vcb.
            m_B (float): B meson mass. It is assumed that this value will never change when handling caches.
            m_V (float): V(ector) meson mass. It is assumed that this value will never change when handling caches.
            m_L (float): Lepton mass. Currently it only limits the kinematic phase-space, i.e. ``self.w_max`` via q2_min = m_L ** 2,
                         but does not affect the differential decay width via a scalar form-factor.
            G_F (float): Effective coupling constant of the weak interaction (Fermi's constant) in units of GeV ** -2. Default value from: https://pdg.lbl.gov/2020/reviews/rpp2020-rev-phys-constants.pdf.
            eta_EW (float): Electroweak corrections.
            BR_Dstar_decay (float, optional): In case the D* meson decay is not treated fully inclusive (BR < 1). It is assumed that this value will never change when handling caches.
        """
        assert 0 <= BR_D_decay <= 1
        self.FF = FF

        self.Vxb = Vxb
        self.mB = FF.m_B if m_B is None else m_B
        self.mV = FF.m_V if m_P is None else m_P
        self.mL = FF.m_L if m_L is None else m_L
        self.GF = G_F
        self.eta_EW = eta_EW
        self.BR_D_decay = BR_D_decay

        # Boundaries of the 4D rate. These assumptions are imposed in the analytical integrations in Mathematica.
        self.kinematics = Kinematics(self.mB, self.mV, self.mL)
        self.w_min, self.w_max = self.kinematics.w_range_numerical_stable
        self.cosL_min, self.cosL_max = self.kinematics.cosL_range

        self.N0 = self.BR_Dstar_decay * self.GF ** 2 / (2*np.pi) ** 4 * self.eta_EW ** 2 / 12 / self.mB ** 2 
        self.N0 *= (2 * self.mB * self.mV)  # Differential dq2/dw 


    def f(self, w):
        q2 = self.kinematics.q2(w)
        p = self.kinematics.p(q2)
        mL = self.mL
        return (q2 - mL ** 2) ** 2 * p / q2


    def dGamma_dw_dcosL(
        self, 
        w: float,
        cosL: float,
        ) -> float:
        """Full 2D differential decay rate for B to P(seudoscalar) meson decays.

        Recommended use cases: 
            * Form factor reweighting.

        Args:
            w (float): Recoil against the hadronic system.
            cosL (float): Lepton angle.
        
        Returns:
            float: Rate at the requested phase space point.
        """
        if not self.w_min <= w <= self.w_max: return 0
        if not self.cosL_min <= cosL <= self.cosL_max: return 0

        mL = self.mL
        q2 = self.kinematics.q2(w)
        Hzero = self.FF.Hzero(w)
        Hscalar = self.FF.Hscalar(w)

        return -3 * np.pi * self.f(w) * self.N0 / 2 / q2 * (
            (-1 + cosL ** 2) * q2 * Hzero ** 2
            - mL * (-cosL * Hzero + Hscalar) ** 2
        )


    def DGamma_Dw_DcosL(
        self,
        wmin: float = None,
        wmax: float = None,
        cosLmin: float = None,
        cosLmax: float = None,
        ) -> float:
        """The differential decay rate for B to P(seudoscalar) meson decays, where the angular variables are analytically integrated.
        
        The integration over w is performed numerically, because analyitcal integration would introduce a dependency on the chosen form factor parametrization.

        Recommended use case:
            * Fitting. The class also provides interfaces to the marginalized distributions directly.

        Args:
            wmin (float): Recoil against the hadronic system lower boundary.
            wmax (float): Recoil against the hadronic system upper boundary.
            cosLmin (float): Lepton angle lower boundary.
            cosLmax (float): Lepton angle upper boundary.

        Returns:
            float: Rate in the marginalized region of the phase space.
        """
        wmin = self.w_min if wmin is None else wmin
        wmax = self.w_max if wmax is None else wmax
        cosLmin = self.cosL_min if cosLmin is None else cosLmin
        cosLmax = self.cosL_max if cosLmax is None else cosLmax

        assert self.w_min <= wmin < wmax <= self.w_max, f"{wmin}, {wmax}"
        assert self.cosL_min <= cosLmin < cosLmax <= self.cosL_max

        mL = self.mL
        q2 = lambda w: self.kinematics.q2(w)
        Hzero = lambda w: self.FF.Hzero(w)
        Hscalar = lambda w: self.FF.Hscalar(w)

        return quad(lambda w: 1 / q2(w) * (cosLmax - cosLmin) * np.pi * self.f(w) * self.N0 * (
            -((-3 + cosLmax ** 2 + cosLmax * cosLmin + cosLmin **2) * q2(w) * Hzero(w) ** 2)
            + mL ** 2 * (
            (cosLmax ** 2 + cosLmax * cosLmin + cosLmin ** 2) * Hzero(w) ** 2 
            - 3 * (cosLmax + cosLmin) * Hzero(w) * Hscalar(w)
            + 3 * Hscalar(w) ** 2
            )
        ), wmin, wmax)[0]


    def Gamma(self) -> float:
        return self.DGamma_Dw_DcosL(self.w_min, self.w_max, self.cosL_min, self.cosL_max)


    def DGamma_Dw(self, wmin: float, wmax: float) -> float:
        return self.DGamma_Dw_DcosL(wmin, wmax, self.cosL_min, self.cosL_max)


    def DGamma_DcosL(self, cosLmin: float, cosLmax: float) -> float:
        return self.DGamma_Dw_DcosL(self.w_min, self.w_max, cosLmin, cosLmax)


    def dGamma_dw(self, w: float) -> float:
        assert self.w_min <= w <= self.w_max
        mL = self.mL
        q2 = self.kinematics.q2(w)
        Hzero = self.FF.Hzero(w)
        Hscalar = self.FF.Hscalar(w)

        return np.pi * self.f(w) * self.N0 / q2 * (
            2 * q2 * Hzero ** 2
            + mL * (Hzero ** 2 + 3 * Hscalar ** 2)
        )


    def dGamma_dcosL(self, cosL: float) -> float:
        assert self.cosL_min <= cosL <= self.cosL_max

        return quad(lambda w: self.dGamma_dw_dcosL(w, cosL), self.w_min, self.w_max)[0]


if __name__ == "__main__":
    pass
