import numpy as np
from effort2.formfactors.kinematics import Kinematics
from effort2.math.integrate import quad


class BtoV:

    def __init__(
        self, 
        FF: None,
        Vxb: float, 
        m_B: float = None, 
        m_V: float = None,
        m_L: float = None,
        G_F: float = 1.1663787e-5,
        eta_EW: float = 1.0066,
        BR_Dstar_decay: float = 1,
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
        assert 0 <= BR_Dstar_decay <= 1
        self.FF = FF

        self.Vxb = Vxb
        self.mB = FF.m_B if m_B is None else m_B
        self.mV = FF.m_V if m_V is None else m_V
        self.mL = FF.m_L if m_L is None else m_L
        self.GF = G_F
        self.eta_EW = eta_EW
        self.BR_Dstar_decay = BR_Dstar_decay

        # Boundaries of the 4D rate. These assumptions are imposed in the analytical integrations in Mathematica.
        self.kinematics = Kinematics(self.mB, self.mV, self.mL)
        self.w_min, self.w_max = self.kinematics.w_range_numerical_stable
        self.cosL_min, self.cosL_max = self.kinematics.cosL_range
        self.cosV_min, self.cosV_max = self.kinematics.cosV_range
        self.chi_min, self.chi_max = self.kinematics.chi_range
                
        self.N0 = self.BR_Dstar_decay * self.GF ** 2 / (2*np.pi) ** 4 * self.eta_EW ** 2 / 12 / self.mB ** 2 
        self.N0 *= (2 * self.mB * self.mV)  # Differential dq2/dw 

        # Helper variable to work with angular coefficients
        self.xi = self.GF ** 2 * self.mB ** 3 / 2 / np.pi ** 4 * self.eta_EW ** 2 * self.BR_Dstar_decay 

        # Sign conventions, do not change.
        self.sign_alpha = +1
        self.sign_beta  = +1


    def f(self, w):
        q2 = self.kinematics.q2(w)
        p = self.kinematics.p(q2)
        mL = self.mL
        return (q2 - mL ** 2) ** 2 * p / q2


    def dGamma_dw_dcosL_dcosV_dchi(
        self, 
        w: float,
        cosL: float,
        cosV: float,
        chi: float
        ) -> float:
        """Full 4D differential decay rate for B to V(ector) meson decays.

        Recommended use cases: 
            * Form factor reweighting.

        Args:
            w (float): Recoil against the hadronic system.
            cosL (float): Lepton angle.
            cosV (float): Vector meson angle.
            chi (float): Decay plane angle.
        
        Returns:
            float: Rate at the requested phase space point.
        """
        if not self.w_min <= w <= self.w_max: return 0
        if not self.cosL_min <= cosL <= self.cosL_max: return 0
        if not self.cosV_min <= cosV <= self.cosV_max: return 0
        if not self.chi_min <= chi <= self.chi_max: return 0

        mL = self.mL
        Hplus = self.FF.Hplus(w)
        Hminus = self.FF.Hminus(w)
        Hzero = self.FF.Hzero(w)
        Hscalar = self.FF.Hscalar(w)

        alpha = self.sign_alpha
        beta = self.sign_beta
        
        return 9 / 32 * self.f(w) * self.N0 * self.Vxb ** 2 * (
            -(-1 + cosV ** 2) * (1 + cosL ** 2 + 2 * cosL * alpha) * Hminus ** 2
            -(-1 + cosV ** 2) * (1 + cosL ** 2 - 2 * cosL * alpha) * Hplus ** 2
            -4 * (1 - cosL ** 2) ** 0.5 * cosV * (1 - cosV ** 2) ** 0.5 * (-cosL + beta) * np.cos(chi) * Hplus * Hzero
            -4 * (1 - cosL ** 2) ** 0.5 * cosV * (1 - cosV ** 2) ** 0.5 * (-cosL - beta) * np.cos(chi) * Hminus * Hzero
            -4 * (-1 + cosL ** 2) * cosV ** 2 * Hzero ** 2
            -2 * (-1 + cosL ** 2) * (-1 + cosV ** 2) * np.cos(2 * chi) * Hplus * Hminus
            + mL ** 2 / self.kinematics.q2(w) #* (  # TODO: Implementation needs to be checked against physics
#                +(-1 + cosL ** 2) * (-1 + cosV ** 2) * Hminus ** 2 
#                +(-1 + cosL ** 2) * (-1 + cosV ** 2) * Hplus ** 2
#                -4 * (1 - cosL ** 2) ** 0.5 * cosV * (1 - cosV ** 2) ** 0.5 * np.cos(chi) * Hplus * (cosL * Hzero - Hscalar)
#                +4 * cosV ** 2 * (-cosL * Hzero + Hscalar) ** 2
#                + Hminus * (
#            (-1 + cosL ** 2) * (-1 + cosV ** 2) * np.cos(2 * chi) * Hplus 
#            + 4 * (1 - cosL ** 2) ** 0.5 * cosV * (1 - cosV ** 2) ** 0.5 * np.cos(chi) * (-cosL * Hzero + Hscalar))
#            ) 
        )
        

    def DGamma_Dw_DcosL_DcosV_Dchi(
        self,
        wmin: float = None,
        wmax: float = None,
        cosLmin: float = None,
        cosLmax: float = None,
        cosVmin: float = None,
        cosVmax: float = None,
        chimin: float = None,
        chimax: float = None,
        ) -> float:
        """The differential decay rate for B to V(ector) meson decays, where the angular variables are analytically integrated.
        
        The integration over w is performed numerically, because analyitcal integration would introduce a dependency on the chosen form factor parametrization.

        Recommended use case:
            * Fitting. The class also provides interfaces to the marginalized distributions directly.

        Args:
            wmin (float): Recoil against the hadronic system lower boundary.
            wmax (float): Recoil against the hadronic system upper boundary.
            cosLmin (float): Lepton angle lower boundary.
            cosLmax (float): Lepton angle upper boundary.
            cosVmin (float): Vector meson angle lower boundary.
            cosVmax (float): Vector meson angle upper boundary.
            chimin (float): Decay plane angle lower boundary.
            chimax (float): Decay plane angle upper boundary.

        Returns:
            float: Rate in the marginalized region of the phase space.
        """
        wmin = self.w_min if wmin is None else wmin
        wmax = self.w_max if wmax is None else wmax
        cosLmin = self.cosL_min if cosLmin is None else cosLmin
        cosLmax = self.cosL_max if cosLmax is None else cosLmax
        cosVmin = self.cosV_min if cosVmin is None else cosVmin
        cosVmax = self.cosV_max if cosVmax is None else cosVmax
        chimin = self.chi_min if chimin is None else chimin
        chimax = self.chi_max if chimax is None else chimax

        assert self.w_min <= wmin < wmax <= self.w_max, f"{wmin}, {wmax}"
        assert self.cosL_min <= cosLmin < cosLmax <= self.cosL_max
        assert self.cosV_min <= cosVmin < cosVmax <= self.cosV_max
        assert self.chi_min <= chimin < chimax <= self.chi_max

        mL = self.mL
        Hplus = lambda w: self.FF.Hplus(w)
        Hminus = lambda w: self.FF.Hminus(w)
        Hzero = lambda w: self.FF.Hzero(w)

        alpha = self.sign_alpha
        beta = self.sign_beta

        return quad(lambda w: -1 / 64 * self.f(w) * self.N0 * self.Vxb ** 2 *  (
            +2 * (-3 * cosLmax + cosLmax ** 3 + 3 * cosLmin - cosLmin ** 3) * (-3 * cosVmax + cosVmax ** 3 + 3 * cosVmin - cosVmin ** 3) * (np.sin(2 * chimax) - np.sin(2 * chimin)) * Hminus(w) * Hplus(w)
            +6 * (chimax - chimin) * (cosLmax - cosLmin) * (cosLmax + cosLmin) * (-3 * cosVmax + cosVmax ** 3 + 3 * cosVmin - cosVmin ** 3) * alpha * (Hminus(w) ** 2 - Hplus(w) ** 2)
            -2 * (chimax - chimin) * (3 * cosLmax + cosLmax ** 3 - cosLmin * (3 + cosLmin ** 2)) * (3 * cosVmax - cosVmax ** 3 - 3 * cosVmin + cosVmin ** 3) * (Hminus(w) ** 2 + Hplus(w) ** 2)
            +12 * (
                -(1 - cosVmax ** 2) ** 0.5 + cosVmax ** 2 * (1 - cosVmax ** 2) ** 0.5 + (1 - cosVmin ** 2) ** 0.5 - cosVmin ** 2 * (1 - cosVmin ** 2) ** 0.5
                ) * beta * (
                    -cosLmax * (1 - cosLmax ** 2) ** 0.5 + cosLmin * (1 - cosLmin ** 2) ** 0.5 + np.arcsin(-cosLmax) - np.arcsin(-cosLmin)
                    ) * (np.sin(chimax) - np.sin(chimin)) * (Hminus(w) - Hplus(w)) * Hzero(w)
            -8 * (
                -(1 - cosLmax ** 2) ** 0.5 + cosLmax ** 2 * (1 - cosLmax ** 2) ** 0.5 + (1 - cosLmin ** 2) ** 0.5 - cosLmin ** 2 * (1 - cosLmin ** 2) ** 0.5
                ) * (
                    -(1 - cosVmax ** 2) ** 0.5 + cosVmax ** 2 * (1 - cosVmax ** 2) ** 0.5 + (1 - cosVmin ** 2) ** 0.5 - cosVmin ** 2 * (1 - cosVmin ** 2) ** 0.5
                    ) * (np.sin(chimax) - np.sin(chimin)) * (Hminus(w) + Hplus(w)) * Hzero(w)
            -8 * (chimax - chimin) * (3 * cosLmax - cosLmax ** 3 - 3 * cosLmin + cosLmin ** 3) * (cosVmax ** 3 - cosVmin ** 3) * Hzero(w) ** 2
            + mL ** 2 / self.kinematics.q2(w) * (
                0 # TODO
            ) 
        ), wmin, wmax)[0]


    def Gamma(self) -> float:
        return self.DGamma_Dw_DcosL_DcosV_Dchi(self.w_min, self.w_max, self.cosL_min, self.cosL_max, self.cosV_min, self.cosV_max, self.chi_min, self.chi_max)


    def DGamma_Dw(self, wmin: float, wmax: float) -> float:
        return self.DGamma_Dw_DcosL_DcosV_Dchi(wmin, wmax, self.cosL_min, self.cosL_max, self.cosV_min, self.cosV_max, self.chi_min, self.chi_max)


    def DGamma_DcosL(self, cosLmin: float, cosLmax: float) -> float:
        return self.DGamma_Dw_DcosL_DcosV_Dchi(self.w_min, self.w_max, cosLmin, cosLmax, self.cosV_min, self.cosV_max, self.chi_min, self.chi_max)


    def DGamma_DcosV(self, cosVmin: float, cosVmax: float) -> float:
        return self.DGamma_Dw_DcosL_DcosV_Dchi(self.w_min, self.w_max, self.cosL_min, self.cosL_max, cosVmin, cosVmax, self.chi_min, self.chi_max)


    def DGamma_Dchi(self, chimin: float, chimax: float) -> float:
        return self.DGamma_Dw_DcosL_DcosV_Dchi(self.w_min, self.w_max, self.cosL_min, self.cosL_max, self.cosV_min, self.cosV_max, chimin, chimax)


    def dGamma_dw(self, w: float) -> float:
        assert self.w_min <= w <= self.w_max
        Hplus = self.FF.Hplus(w)
        Hminus = self.FF.Hminus(w)
        Hzero = self.FF.Hzero(w)
        return -3 / 32 * self.f(w) * self.N0 * self.Vxb ** 2 * (
            - 64 / 3 * np.pi * Hminus ** 2 
            - 64 / 3 * np.pi * Hplus ** 2 
            - 64 / 3 * np.pi * Hzero ** 2
            )


    def dGamma_dcosL(self, cosL: float) -> float:
        assert self.cosL_min <= cosL <= self.cosL_max
        Hplus = lambda w: self.FF.Hplus(w)
        Hminus = lambda w: self.FF.Hminus(w)
        Hzero = lambda w: self.FF.Hzero(w)
        alpha = self.sign_alpha

        return quad(
            lambda w: -3 / 32 * self.f(w) * self.N0 * self.Vxb ** 2 * (
                - 8 * np.pi * (1 + cosL ** 2 + 2 * cosL * alpha) * Hminus(w) ** 2 
                - 8 * np.pi * (1 + cosL ** 2 - 2 * cosL * alpha) * Hplus(w) ** 2 
                + 16 * np.pi * (-1 + cosL ** 2) * Hzero(w) ** 2
                ), self.w_min, self.w_max
                )[0]


    def dGamma_dcosV(self, cosV: float) -> float:
        assert self.cosV_min <= cosV <= self.cosV_max
        Hplus = lambda w: self.FF.Hplus(w)
        Hminus = lambda w: self.FF.Hminus(w)
        Hzero = lambda w: self.FF.Hzero(w)

        return quad(
            lambda w: -3 / 32 * self.f(w) * self.N0 * self.Vxb ** 2 * (
                + 16 * np.pi * (-1 + cosV ** 2) * Hminus(w) ** 2 
                + 16 * np.pi * (-1 + cosV ** 2) * Hplus(w) ** 2 
                - 32 * np.pi * cosV ** 2 * Hzero(w) ** 2
                ),self.w_min, self.w_max
                )[0]


    def dGamma_dchi(self, chi: float) -> float:
        assert self.chi_min <= chi <= self.chi_max
        Hplus = lambda w: self.FF.Hplus(w)
        Hminus = lambda w: self.FF.Hminus(w)
        Hzero = lambda w: self.FF.Hzero(w)
        return quad(
            lambda w: -1 / 32 * self.f(w) * self.N0 * self.Vxb ** 2 * (
                - 32 * Hminus(w) ** 2 
                + 32 * np.cos(2 * chi) * Hminus(w) * Hplus(w) 
                - 32 * Hplus(w) ** 2 
                - 32 * Hzero(w) ** 2
                ), self.w_min, self.w_max
                )[0]


if __name__ == "__main__":
    pass
