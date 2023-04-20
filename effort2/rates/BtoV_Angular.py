import numpy as np
from effort2.formfactors.kinematics import Kinematics
from effort2.math.integrate import quad


class BtoV:

    def __init__(
        self, 
        Angular: None,
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
            Vxb (float): CKM parameter Vxb.
            m_B (float): B meson mass. It is assumed that this value will never change when handling caches.
            m_V (float): V(ector) meson mass. It is assumed that this value will never change when handling caches.
            m_L (float): Lepton mass. Currently it only limits the kinematic phase-space, i.e. ``self.w_max`` via q2_min = m_L ** 2,
                         but does not affect the differential decay width via a scalar form-factor.
            G_F (float): Effective coupling constant of the weak interaction (Fermi's constant) in units of GeV ** -2. Default value from: https://pdg.lbl.gov/2020/reviews/rpp2020-rev-phys-constants.pdf.
            eta_EW (float): Electroweak corrections.
            BR_Dstar_decay (float, optional): In case the D* meson decay is not treated fully inclusive (BR < 1). It is assumed that this value will never change when handling caches.
        """
        assert 0 <= BR_Dstar_decay <= 1
        self.Angular = Angular

        self.Vxb = Vxb
        self.mB = Angular.FF.m_B if m_B is None else m_B
        self.mV = Angular.FF.m_V if m_V is None else m_V
        self.mL = Angular.FF.m_L if m_L is None else m_L
        self.GF = G_F
        self.eta_EW = eta_EW
        self.BR_Dstar_decay = BR_Dstar_decay

        # Boundaries of the 4D rate. These assumptions are imposed in the analytical integrations in Mathematica.
        self.kinematics = Kinematics(self.mB, self.mV, self.mL)
        self.w_min, self.w_max = self.kinematics.w_range_numerical_stable
        self.cosL_min, self.cosL_max = self.kinematics.cosL_range
        self.cosV_min, self.cosV_max = self.kinematics.cosV_range
        self.chi_min, self.chi_max = self.kinematics.chi_range
                
        self.A = self.BR_Dstar_decay * self.GF ** 2 / 2 / np.pi ** 4 * self.eta_EW ** 2 * self.mB ** 3
        self.A *= (2 * self.mB * self.mV)  # Differential dq2/dw 


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

        q2 = self.kinematics.q2(w)
        J1s = self.Angular.J1s(q2)
        J1c = self.Angular.J1c(q2)
        J2s = self.Angular.J2s(q2)
        J2c = self.Angular.J2c(q2)
        J3  = self.Angular.J3(q2)
        J4  = self.Angular.J4(q2)
        J5  = self.Angular.J5(q2)
        J6s = self.Angular.J6s(q2)
        J6c = self.Angular.J6c(q2)
        J7  = self.Angular.J7(q2)
        J8  = self.Angular.J8(q2)
        J9  = self.Angular.J9(q2)
        
        return self.A * self.Vxb ** 2 * (
            (1 - cosV ** 2) * J1s + cosV ** 2 * J1c
            + (-1 + 2 * cosL ** 2) * (cosV ** 2 * J2c + (1 - cosV ** 2) * J2s)
            + cosL * (cosV ** 2 * J6c + (1 - cosV ** 2) * J6s)
            + 4 * cosL * (1 - cosL ** 2) ** 0.5 * cosV * (1 - cosV ** 2) ** 0.5 * J4 * np.cos(chi)
            + 2 * (1 - cosL ** 2) ** 0.5 * cosV * (1 - cosV ** 2) * J5 * np.cos(chi)
            + (1 - cosL ** 2) * (1 - cosV ** 2) * J3 * np.cos(2 * chi)
            + 2 * np.sqrt(1 - cosL ** 2) * cosV * np.sqrt(1 - cosV ** 2) * J7 * np.sin(chi)
            + 4 * cosL * np.sqrt(1 - cosL ** 2) * cosV * np.sqrt(1 - cosV ** 2) * J8 * np.sin(chi)
            + (1 - cosL ** 2) * (1 - cosV ** 2) * J9 * np.sin(2 * chi)
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

        J1s = lambda w: self.Angular.J1s(self.kinematics.q2(w))
        J1c = lambda w: self.Angular.J1c(self.kinematics.q2(w))
        J2s = lambda w: self.Angular.J2s(self.kinematics.q2(w))
        J2c = lambda w: self.Angular.J2c(self.kinematics.q2(w))
        J3  = lambda w: self.Angular.J3(self.kinematics.q2(w))
        J4  = lambda w: self.Angular.J4(self.kinematics.q2(w))
        J5  = lambda w: self.Angular.J5(self.kinematics.q2(w))
        J6s = lambda w: self.Angular.J6s(self.kinematics.q2(w))
        J6c = lambda w: self.Angular.J6c(self.kinematics.q2(w))
        J7  = lambda w: self.Angular.J7(self.kinematics.q2(w))
        J8  = lambda w: self.Angular.J8(self.kinematics.q2(w))
        J9  = lambda w: self.Angular.J9(self.kinematics.q2(w))

        rate = quad(lambda w: 1 / 18 * self.A * self.Vxb ** 2 * (
            J1c(w) * 6 * (
            + chimax * cosLmax * cosVmax ** 3 - chimin * cosLmax * cosVmax ** 3 - chimax * cosLmin * cosVmax ** 3 + chimin * cosLmin * cosVmax ** 3 
            - chimax * cosLmax * cosVmin ** 3 + chimin * cosLmax * cosVmin ** 3 + chimax * cosLmin * cosVmin ** 3 - chimin * cosLmin * cosVmin ** 3
            ) + J1s(w) * 6 * (
            + 3 * chimax * cosLmax * cosVmax - 3 * chimin * cosLmax * cosVmax - 3 * chimax * cosLmin * cosVmax +  3 * chimin * cosLmin * cosVmax 
            - chimax * cosLmax * cosVmax ** 3 + chimin * cosLmax * cosVmax ** 3 + chimax * cosLmin * cosVmax ** 3 - chimin * cosLmin * cosVmax ** 3 
            - 3 * chimax * cosLmax * cosVmin + 3 * chimin * cosLmax * cosVmin + 3 * chimax * cosLmin * cosVmin -  3 * chimin * cosLmin * cosVmin 
            + chimax * cosLmax * cosVmin ** 3 - chimin * cosLmax * cosVmin ** 3 - chimax * cosLmin * cosVmin ** 3 + chimin * cosLmin * cosVmin ** 3
            ) + J2c(w) * (
            - 6 * chimax * cosLmax * cosVmax ** 3 + 6 * chimin * cosLmax * cosVmax ** 3 + 4 * chimax * cosLmax ** 3 * cosVmax ** 3 - 4 * chimin * cosLmax ** 3 * cosVmax ** 3 
            + 6 * chimax * cosLmin * cosVmax ** 3 - 6 * chimin * cosLmin * cosVmax ** 3 - 4 * chimax * cosLmin ** 3 * cosVmax ** 3 + 4 * chimin * cosLmin ** 3 * cosVmax ** 3 
            + 6 * chimax * cosLmax * cosVmin ** 3 - 6 * chimin * cosLmax * cosVmin ** 3 - 4 * chimax * cosLmax ** 3 * cosVmin ** 3 + 4 * chimin * cosLmax ** 3 * cosVmin ** 3 
            - 6 * chimax * cosLmin * cosVmin ** 3 + 6 * chimin * cosLmin * cosVmin ** 3 + 4 * chimax * cosLmin ** 3 * cosVmin ** 3 - 4 * chimin * cosLmin ** 3 * cosVmin ** 3 
            ) + J2s(w) * (
            - 18 * chimax * cosLmax * cosVmax + 18 * chimin * cosLmax * cosVmax + 12 * chimax * cosLmax ** 3 * cosVmax - 12 * chimin * cosLmax ** 3 * cosVmax
            + 18 * chimax * cosLmin * cosVmax - 18 * chimin * cosLmin * cosVmax - 12 * chimax * cosLmin ** 3 * cosVmax + 12 * chimin * cosLmin ** 3 * cosVmax
            + 6 * chimax * cosLmax * cosVmax ** 3 - 6 * chimin * cosLmax * cosVmax ** 3 - 4 * chimax * cosLmax ** 3 * cosVmax ** 3 + 4 * chimin * cosLmax ** 3 * cosVmax ** 3
            - 6 * chimax * cosLmin * cosVmax ** 3 + 6 * chimin * cosLmin * cosVmax ** 3 + 4 * chimax * cosLmin ** 3 * cosVmax ** 3 - 4 * chimin * cosLmin ** 3 * cosVmax ** 3
            + 18 * chimax * cosLmax * cosVmin - 18 * chimin * cosLmax * cosVmin - 12 * chimax * cosLmax ** 3 * cosVmin + 12 * chimin * cosLmax ** 3 * cosVmin
            - 18 * chimax * cosLmin * cosVmin + 18 * chimin * cosLmin * cosVmin + 12 * chimax * cosLmin ** 3 * cosVmin - 12 * chimin * cosLmin ** 3 * cosVmin
            - 6 * chimax * cosLmax * cosVmin ** 3 + 6 * chimin * cosLmax * cosVmin ** 3 + 4 * chimax * cosLmax ** 3 * cosVmin ** 3 - 4 * chimin * cosLmax ** 3 * cosVmin ** 3
            + 6 * chimax * cosLmin * cosVmin ** 3 - 6 * chimin * cosLmin * cosVmin ** 3 - 4 * chimax * cosLmin ** 3 * cosVmin ** 3 + 4 * chimin * cosLmin ** 3 * cosVmin ** 3
            ) + J6c(w) * 3 * (
            + chimax * cosLmax ** 2 * cosVmax ** 3 - chimin * cosLmax ** 2 * cosVmax ** 3 - chimax * cosLmin ** 2 * cosVmax ** 3 + chimin * cosLmin ** 2 * cosVmax ** 3 
            - chimax * cosLmax ** 2 * cosVmin ** 3 + chimin * cosLmax ** 2 * cosVmin ** 3 + chimax * cosLmin ** 2 * cosVmin ** 3 - chimin * cosLmin ** 2 * cosVmin ** 3
            ) + J6s(w) * 3 * (
            + 3 * chimax * cosLmax ** 2 * cosVmax - 3 * chimin * cosLmax ** 2 * cosVmax - 3 * chimax * cosLmin ** 2 * cosVmax + 3 * chimin * cosLmin ** 2 * cosVmax
            - chimax * cosLmax ** 2 * cosVmax ** 3 + chimin * cosLmax ** 2 * cosVmax ** 3 + chimax * cosLmin ** 2 * cosVmax ** 3 - chimin * cosLmin ** 2 * cosVmax ** 3
            - 3 * chimax * cosLmax ** 2 * cosVmin + 3 * chimin * cosLmax ** 2 * cosVmin + 3 * chimax * cosLmin ** 2 * cosVmin - 3 * chimin * cosLmin ** 2 * cosVmin
            + chimax * cosLmax ** 2 * cosVmin ** 3 - chimin * cosLmax ** 2 * cosVmin ** 3 - chimax * cosLmin ** 2 * cosVmin ** 3 + chimin * cosLmin ** 2 * cosVmin ** 3
            ) + J7(w) * 6 
            * ((1 - cosVmax ** 2) ** 0.5 - cosVmax ** 2 * (1 - cosVmax ** 2) ** 0.5 - (1 - cosVmin ** 2) ** 0.5 + cosVmin ** 2 * (1 - cosVmin ** 2) ** 0.5) 
            * (cosLmax * (1 - cosLmax ** 2) ** 0.5 - cosLmin * (1 - cosLmin ** 2) ** 0.5 + np.arcsin(cosLmax) - np.arcsin(cosLmin)) 
            * (np.cos(chimax) - np.cos(chimin)) 
            + J8(w) * -8 
            * (-(1 - cosLmax ** 2) ** 0.5 + cosLmax ** 2 * (1 - cosLmax ** 2) ** 0.5 + (1 - cosLmin ** 2) - cosLmin ** 2 * (1 - cosLmin ** 2) **0.5) 
            * (-(1 - cosVmax ** 2) ** 0.5 + cosVmax ** 2 * (1 - cosVmax ** 2) ** 0.5 + (1 - cosVmin ** 2) ** 0.5 - cosVmin ** 2 * (1 - cosVmin ** 2) ** 0.5) 
            * (np.cos(chimax) - np.cos(chimin))
            + J9(w) * (
            + 3 * (cosLmax ** 3 - cosLmin ** 3) * (cosVmax - cosVmin) * (np.cos(2 * chimax) - np.cos(2 * chimin)) 
            + 3 * (cosLmax - cosLmin) * (cosVmax ** 3 - cosVmin ** 3) * (np.cos(2 * chimax) - np.cos(2 * chimin))
            + 9 * (cosLmax - cosLmin) * (cosVmax - cosVmin) * (-np.cos(2 * chimax) + np.cos(2 * chimin)) 
            + (cosLmax ** 3 - cosLmin ** 3) * (cosVmax ** 3 - cosVmin ** 3) * (-np.cos(2 * chimax) + np.cos(2 * chimin))
            ) 
            + J4(w) * 8 
            * (-(1 - cosLmax ** 2) ** 0.5 + cosLmax ** 2 * (1 - cosLmax ** 2) ** 0.5 + (1 - cosLmin ** 2) ** 0.5 - cosLmin ** 2 * (1 - cosLmin ** 2) ** 0.5)
            * (-(1 - cosVmax ** 2) ** 0.5 + cosVmax ** 2 * (1 - cosVmax ** 2) ** 0.5 + (1 - cosVmin ** 2) ** 0.5 - cosVmin ** 2 * (1 - cosVmin ** 2) ** 0.5) 
            * (np.sin(chimax) - np.sin(chimin))
            + J5(w) * 6
            * (-(1 - cosVmax ** 2) ** 0.5 + cosVmax ** 2 * (1 - cosVmax ** 2) ** 0.5 + (1 - cosVmin ** 2) ** 0.5 - cosVmin ** 2 * (1 - cosVmin ** 2) ** 0.5) 
            * (cosLmax * (1 - cosLmax ** 2) ** 0.5 - cosLmin * (1 - cosLmin ** 2) ** 0.5 + np.arcsin(cosLmax) - np.arcsin(cosLmin)) 
            * (np.sin(chimax) - np.sin(chimin))
            + J3(w) * (
            + 9 * (cosLmax - cosLmin) * (cosVmax - cosVmin) * (np.sin(2 * chimax) - np.sin(2 * chimin)) 
            - 3 * (cosLmax ** 3 - cosLmin ** 3) * (cosVmax - cosVmin) * (np.sin(2 * chimax) - np.sin(2 * chimin)) 
            - 3 * (cosLmax - cosLmin) * (cosVmax ** 3 - cosVmin ** 3) * (np.sin(2 * chimax) - np.sin(2 * chimin)) 
            + (cosLmax ** 3 - cosLmin ** 3) * (cosVmax ** 3 - cosVmin ** 3) * (np.sin(2 * chimax) - np.sin(2 * chimin))
            )
            ), wmin, wmax)[0]

        return rate


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
        q2 = self.kinematics.q2(w)
        J1s = self.Angular.J1s(q2)
        J1c = self.Angular.J1c(q2)
        J2s = self.Angular.J2s(q2)
        J2c = self.Angular.J2c(q2)
        rate = 8 / 9 * np.pi * self.A * self.Vxb ** 2 * (
            3 * J1c + 6 * J1s - J2c - 2 *J2s
            ) 
        return rate


    def dGamma_dcosL(self, cosL: float) -> float:
        assert self.cosL_min <= cosL <= self.cosL_max
        J1s = lambda w: self.Angular.J1s(self.kinematics.q2(w))
        J1c = lambda w: self.Angular.J1c(self.kinematics.q2(w))
        J2s = lambda w: self.Angular.J2s(self.kinematics.q2(w))
        J2c = lambda w: self.Angular.J2c(self.kinematics.q2(w))
        J6s = lambda w: self.Angular.J6s(self.kinematics.q2(w))
        J6c = lambda w: self.Angular.J6c(self.kinematics.q2(w))
        rate = quad(
            lambda w: 4 / 3 * np.pi * self.A * self.Vxb ** 2 * (
            J1c(w) + 2 * J1s(w) - J2c(w) + 2 * cosL ** 2 * J2c(w) - 2 * J2s(w) + 4 * cosL ** 2 * J2s(w) + cosL * J6c(w) + 2 * cosL * J6s(w)
            ), self.w_min, self.w_max
            )[0]
        return rate


    def dGamma_dcosV(self, cosV: float) -> float:
        assert self.cosV_min <= cosV <= self.cosV_max
        J1s = lambda w: self.Angular.J1s(self.kinematics.q2(w))
        J1c = lambda w: self.Angular.J1c(self.kinematics.q2(w))
        J2s = lambda w: self.Angular.J2s(self.kinematics.q2(w))
        J2c = lambda w: self.Angular.J2c(self.kinematics.q2(w))
        rate = quad(
            lambda w: 4 / 3 * np.pi * self.A * self.Vxb ** 2 * (
            3 * cosV ** 2 * J1c(w) - 3 * (-1 + cosV ** 2) * J1s(w) - cosV ** 2 * J2c(w) - J2s(w) + cosV ** 2 * J2s(w)
            ), self.w_min, self.w_max
            )[0]
        return rate
    

    def dGamma_dchi(self, chi: float) -> float:
        assert self.chi_min <= chi <= self.chi_max
        J1s = lambda w: self.Angular.J1s(self.kinematics.q2(w))
        J1c = lambda w: self.Angular.J1c(self.kinematics.q2(w))
        J2s = lambda w: self.Angular.J2s(self.kinematics.q2(w))
        J2c = lambda w: self.Angular.J2c(self.kinematics.q2(w))
        J3  = lambda w: self.Angular.J3(self.kinematics.q2(w))
        J9  = lambda w: self.Angular.J9(self.kinematics.q2(w))
        rate = quad(
            lambda w: 4 / 9 * self.A * self.Vxb ** 2 * (
            3 * J1c(w) + 6 * J1s(w) - J2c(w) - 2 * J2s(w) + 4 * J3(w) * np.cos(2 * chi) + 4 * J9(w) * np.sin(2 * chi)
            ), self.w_min, self.w_max
            )[0]
        return rate


if __name__ == "__main__":
    pass
