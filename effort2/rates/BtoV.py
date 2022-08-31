import numpy as np
from effort2.formfactors.kinematics import Kinematics
from effort2.math.integrate import quad


class BtoV:

    def __init__(
        self, 
        FF: None,
        Vcb: float, 
        m_B: float = None, 
        m_V: float = None,
        m_L: float = 0,
        G_F: float = 1.1663787e-5,
        eta_EW: float = 1.0066,
        BR_Dstar_decay: float = 1,
        VminusA: float = 1,
        VplusA: float = 0
        ):
        """Initialize a class for calculating decay rate of B to Vector meson decays.

        TODO: 
            * Full lepton mass effects.
            * Add Glebsch Gordan coefficient to differ between B+ and B0 decays.
            * Test the pre integrated rate vs numerical integration of the full rate.
            * Refactor out MC generation into a separate class which gets initialized instead of re-implementing it for every potential rate class.
            * Implement switch to either use scipy/numpy default or enable some extra features to support propagation of uncertainties for the cost of extra run time.

        Nota bene:
            * If you use a form factor parametrization where you can/want to absorb eta_EW and/or Vcb into the definition of the form factor, initialize this class setting both values as 1.
            * The class does not provide marginalizations over w, because this would make the analytical integration dependent on the chosen form factor parametrization. Numerical integration over w is therefore required in many applications.
            * Using the exact boundaries of w on the integration process might cause issues. Try adding/subtracting an epsilon to wmin/wmax to resolve the issue.

        Args:
            Vcb (float): CKM parameter Vcb.
            m_B (float): B meson mass. It is assumed that this value will never change when handling caches.
            m_V (float): V(ector) meson mass. It is assumed that this value will never change when handling caches.
            m_L (float): Lepton mass. Currently it only limits the kinematic phase-space, i.e. ``self.w_max`` via q²_min = m_L²,
                         but does not affect the differential decay width via a scalar form-factor.
            G_F (float): Effective coupling constant of the weak interaction (Fermi's constant) in units of GeV^-2. Default value from: https://pdg.lbl.gov/2020/reviews/rpp2020-rev-phys-constants.pdf.
            eta_EW (float): Electroweak corrections.
            BR_Dstar_decay (float, optional): In case the D* meson decay is not treated fully inclusive (BR < 1). It is assumed that this value will never change when handling caches.
        """
        assert 0 <= BR_Dstar_decay <= 1
        self.FF = FF

        self.Vcb = Vcb
        self.mB = FF.m_B if m_B is None else m_B
        self.mV = FF.m_V if m_V is None else m_V
        self.mL = m_L
        self.GF = G_F
        self.eta_EW = eta_EW
        self.BR_Dstar_decay = BR_Dstar_decay

        # Boundaries of the 4D rate. These assumptions are imposed in the analytical integrations in Mathematica.
        self.kinematics = Kinematics(self.mB, self.mV, self.mL)
        self.w_min, self.w_max = self.kinematics.w_range_numerical_stable
        self.cosL_min, self.cosL_max = self.kinematics.cosL_range
        self.cosV_min, self.cosV_max = self.kinematics.cosV_range
        self.chi_min, self.chi_max = self.kinematics.chi_range
        
        # These are constant factors which turn up in each rate calculation. Let us do it once and cache the result.
        self.r = self.mV / self.mB 
        self.N0 = 6 * self.mB * self.mV ** 2 / 8 / (4*np.pi) ** 4 * self.GF ** 2 * self.eta_EW ** 2 * self.BR_Dstar_decay #* 1e10

        self.VminusA = VminusA
        self.VplusA = VplusA
        # These are required for the generator feature.
        # self.rate_max = self.dGamma_dw_dcosL_dcosV_dchi(*self.dGamma_max())  # Add 10% on top just to be sure.



    def _q2(self, w):
        return self.mB**2 + self.mV**2 - 2 * w * self.mB * self.mV

    def f(self, w):
        return (1 - 2 * w * self.r + self.r ** 2) * (w ** 2 - 1) ** 0.5 * (1 - self.mL**2/self._q2(w))**2

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
        # TODO add (1 - ml^2)^2/q2 factor

        Hplus = self.FF.Hplus(w)
        Hminus = self.FF.Hminus(w)
        Hzero = self.FF.Hzero(w)
        rate_VminusA =  self.f(w) * self.N0 * self.Vcb ** 2 * (
            (1 + cosL) ** 2 * (1 - cosV ** 2) * Hminus ** 2
            - 2 * (1 - cosL ** 2) * (1 - cosV ** 2) * np.cos(2 * chi) * Hminus * Hplus
            + (1 - cosL) ** 2 * (1 - cosV ** 2) * Hplus ** 2
            + 4 * (1 + cosL) * (1 - cosL ** 2) ** 0.5 * cosV * (1 - cosV ** 2) ** 0.5 * np.cos(chi) * Hminus * Hzero
            - 4 * (1 - cosL) * (1 - cosL ** 2) ** 0.5 * cosV * (1 - cosV ** 2) ** 0.5 * np.cos(chi) * Hplus * Hzero
            + 4 * (1 - cosL ** 2) * cosV ** 2 * Hzero ** 2
        )

        # Instead of modfying the rate, we just flip the Helicity amplitude definitions
        Hplus = -self.FF.Hminus(w)
        Hminus = -self.FF.Hplus(w)
        Hzero = -self.FF.Hzero(w)
        rate_VplusA =  self.f(w) * self.N0 * self.Vcb ** 2 * (
            (1 + cosL) ** 2 * (1 - cosV ** 2) * Hminus ** 2
            - 2 * (1 - cosL ** 2) * (1 - cosV ** 2) * np.cos(2 * chi) * Hminus * Hplus
            + (1 - cosL) ** 2 * (1 - cosV ** 2) * Hplus ** 2
            + 4 * (1 + cosL) * (1 - cosL ** 2) ** 0.5 * cosV * (1 - cosV ** 2) ** 0.5 * np.cos(chi) * Hminus * Hzero
            - 4 * (1 - cosL) * (1 - cosL ** 2) ** 0.5 * cosV * (1 - cosV ** 2) ** 0.5 * np.cos(chi) * Hplus * Hzero
            + 4 * (1 - cosL ** 2) * cosV ** 2 * Hzero ** 2
        )

        return self.VminusA * rate_VminusA + self.VplusA * rate_VplusA


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

        Hplus = lambda w: self.FF.Hplus(w)
        Hminus = lambda w: self.FF.Hminus(w)
        Hzero = lambda w: self.FF.Hzero(w)
        rate_VminusA = quad(lambda w: 1 / 3. * self.f(w) * self.N0 * self.Vcb ** 2 * (
            - ( (chimax - chimin) * (cosLmax + cosLmax ** 2 + cosLmax ** 3 / 3 - 1 / 3. * cosLmin * (3 + cosLmin * (3 + cosLmin))) * (-3 * cosVmax + cosVmax ** 3 + 3 * cosVmin - cosVmin ** 3) * Hminus(w) ** 2 )
            - 1 / 3. * (-3 * cosLmax + cosLmax ** 3 + 3 * cosLmin - cosLmin ** 3) * (-3 * cosVmax + cosVmax ** 3 + 3 * cosVmin - cosVmin ** 3) * (np.sin(2 * chimax) - np.sin(2 * chimin)) * Hminus(w) * Hplus(w)
            - 1 / 3. * (chimax - chimin) * (cosLmax - cosLmin) * (3 + cosLmax ** 2 + cosLmax * (-3 + cosLmin) + (-3 + cosLmin) * cosLmin) * (-3 * cosVmax + cosVmax ** 3 + 3 * cosVmin - cosVmin ** 3) * Hplus(w) ** 2
            + 2 / 3. * (-(1 - cosVmax ** 2) ** 0.5 + cosVmax ** 2 * (1 - cosVmax ** 2) ** 0.5 + (1 - cosVmin ** 2) ** 0.5 - cosVmin ** 2 * (1 - cosVmin ** 2) ** 0.5) 
                * (-2 * (1 - cosLmax ** 2) ** 0.5 
                    + 3 * cosLmax * (1 - cosLmax ** 2) ** 0.5
                    + 2 * cosLmax ** 2 * (1 - cosLmax ** 2) ** 0.5 
                    + 2 * (1 - cosLmin ** 2) ** 0.5 
                    - 3 * cosLmin * (1 - cosLmin ** 2) ** 0.5
                    - 2 * cosLmin ** 2 * (1 - cosLmin ** 2) ** 0.5
                    + 3 * np.arcsin(cosLmax)
                    - 3 * np.arcsin(cosLmin)
                ) * (np.sin(chimax) - np.sin(chimin)) * Hminus(w) * Hzero(w)
            + 2 / 3. * (-(1 - cosVmax ** 2) ** 0.5 + cosVmax ** 2 * (1 - cosVmax ** 2) ** 0.5 + (1 - cosVmin ** 2) ** 0.5 - cosVmin ** 2 * (1 - cosVmin ** 2) ** 0.5)
                * (-2 * (1 - cosLmax ** 2) ** 0.5
                    + cosLmax * (-3 + 2 * cosLmax) * (1 - cosLmax ** 2) ** 0.5
                    + 2 * (1 - cosLmin ** 2) ** 0.5
                    + (3 - 2 * cosLmin) * cosLmin * (1 - cosLmin ** 2) ** 0.5
                    - 3 * np.arcsin(cosLmax)
                    + 3 * np.arcsin(cosLmin)
                ) * (np.sin(chimax) - np.sin(chimin)) * Hplus(w) * Hzero(w)
            - 4 / 3. * (chimax - chimin) * (-3 * cosLmax + cosLmax ** 3 + 3 * cosLmin - cosLmin ** 3) * (cosVmax ** 3 - cosVmin ** 3) * Hzero(w) ** 2
        ), wmin, wmax)


        # Instead of modfying the rate, we just flip the Helicity amplitude definitions
        Hplus = lambda w: -self.FF.Hminus(w)
        Hminus = lambda w: -self.FF.Hplus(w)
        Hzero = lambda w: -self.FF.Hzero(w)
        rate_VplusA = quad(lambda w: 1 / 3. * self.f(w) * self.N0 * self.Vcb ** 2 * (
            - ( (chimax - chimin) * (cosLmax + cosLmax ** 2 + cosLmax ** 3 / 3 - 1 / 3. * cosLmin * (3 + cosLmin * (3 + cosLmin))) * (-3 * cosVmax + cosVmax ** 3 + 3 * cosVmin - cosVmin ** 3) * Hminus(w) ** 2 )
            - 1 / 3. * (-3 * cosLmax + cosLmax ** 3 + 3 * cosLmin - cosLmin ** 3) * (-3 * cosVmax + cosVmax ** 3 + 3 * cosVmin - cosVmin ** 3) * (np.sin(2 * chimax) - np.sin(2 * chimin)) * Hminus(w) * Hplus(w)
            - 1 / 3. * (chimax - chimin) * (cosLmax - cosLmin) * (3 + cosLmax ** 2 + cosLmax * (-3 + cosLmin) + (-3 + cosLmin) * cosLmin) * (-3 * cosVmax + cosVmax ** 3 + 3 * cosVmin - cosVmin ** 3) * Hplus(w) ** 2
            + 2 / 3. * (-(1 - cosVmax ** 2) ** 0.5 + cosVmax ** 2 * (1 - cosVmax ** 2) ** 0.5 + (1 - cosVmin ** 2) ** 0.5 - cosVmin ** 2 * (1 - cosVmin ** 2) ** 0.5) 
                * (-2 * (1 - cosLmax ** 2) ** 0.5 
                    + 3 * cosLmax * (1 - cosLmax ** 2) ** 0.5
                    + 2 * cosLmax ** 2 * (1 - cosLmax ** 2) ** 0.5 
                    + 2 * (1 - cosLmin ** 2) ** 0.5 
                    - 3 * cosLmin * (1 - cosLmin ** 2) ** 0.5
                    - 2 * cosLmin ** 2 * (1 - cosLmin ** 2) ** 0.5
                    + 3 * np.arcsin(cosLmax)
                    - 3 * np.arcsin(cosLmin)
                ) * (np.sin(chimax) - np.sin(chimin)) * Hminus(w) * Hzero(w)
            + 2 / 3. * (-(1 - cosVmax ** 2) ** 0.5 + cosVmax ** 2 * (1 - cosVmax ** 2) ** 0.5 + (1 - cosVmin ** 2) ** 0.5 - cosVmin ** 2 * (1 - cosVmin ** 2) ** 0.5)
                * (-2 * (1 - cosLmax ** 2) ** 0.5
                    + cosLmax * (-3 + 2 * cosLmax) * (1 - cosLmax ** 2) ** 0.5
                    + 2 * (1 - cosLmin ** 2) ** 0.5
                    + (3 - 2 * cosLmin) * cosLmin * (1 - cosLmin ** 2) ** 0.5
                    - 3 * np.arcsin(cosLmax)
                    + 3 * np.arcsin(cosLmin)
                ) * (np.sin(chimax) - np.sin(chimin)) * Hplus(w) * Hzero(w)
            - 4 / 3. * (chimax - chimin) * (-3 * cosLmax + cosLmax ** 3 + 3 * cosLmin - cosLmin ** 3) * (cosVmax ** 3 - cosVmin ** 3) * Hzero(w) ** 2
        ), wmin, wmax)

        return self.VminusA * rate_VminusA[0] + self.VplusA * rate_VplusA[0]


    def Gamma(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.DGamma_Dw_DcosL_DcosV_Dchi(self.w_min, self.w_max, self.cosL_min, self.cosL_max, self.cosV_min, self.cosV_max, self.chi_min, self.chi_max)


    def DGamma_Dw(self, wmin: float, wmax: float) -> float:
        """[summary]

        Recommended use cases:
            * Fitting.

        Args:
            wmin (float): [description]
            wmax (float): [description]

        Returns:
            float: [description]
        """
        return self.DGamma_Dw_DcosL_DcosV_Dchi(wmin, wmax, self.cosL_min, self.cosL_max, self.cosV_min, self.cosV_max, self.chi_min, self.chi_max)


    def DGamma_DcosL(self, cosLmin: float, cosLmax: float) -> float:
        """[summary]

        Recommended use cases:
            * Fitting.

        Args:
            cosLmin (float): [description]
            cosLmax (float): [description]

        Returns:
            float: [description]
        """
        return self.DGamma_Dw_DcosL_DcosV_Dchi(self.w_min, self.w_max, cosLmin, cosLmax, self.cosV_min, self.cosV_max, self.chi_min, self.chi_max)


    def DGamma_DcosV(self, cosVmin: float, cosVmax: float) -> float:
        """[summary]

        Recommended use cases:
            * Fitting.

        Args:
            cosVmin (float): [description]
            cosVmax (float): [description]

        Returns:
            float: [description]
        """
        return self.DGamma_Dw_DcosL_DcosV_Dchi(self.w_min, self.w_max, self.cosL_min, self.cosL_max, cosVmin, cosVmax, self.chi_min, self.chi_max)


    def DGamma_Dchi(self, chimin: float, chimax: float) -> float:
        """[summary]
        
        Recommended use cases:
            * Fitting.

        Args:
            chimin (float): [description]
            chimax (float): [description]

        Returns:
            float: [description]
        """
        return self.DGamma_Dw_DcosL_DcosV_Dchi(self.w_min, self.w_max, self.cosL_min, self.cosL_max, self.cosV_min, self.cosV_max, chimin, chimax)


    def dGamma_dw(self, w: float) -> float:
        """[summary]

        Nota bene:
            * Not fully generalized (yet), cosL, cosV and chi are integrated over the full range.

        Recommended use cases:
            * Plotting.

        Args:
            w (float): [description]

        Returns:
            float: [description]
        """
        assert self.w_min <= w <= self.w_max
        Hplus = self.FF.Hplus(w)
        Hminus = self.FF.Hminus(w)
        Hzero = self.FF.Hzero(w)
        rate_VminusA =  self.f(w) / 3 * self.N0 * self.Vcb ** 2 * (64 / 3. * np.pi * Hminus ** 2 + 64 / 3. * np.pi * Hplus ** 2 + 64 / 3. * np.pi * Hzero ** 2)

        Hplus = -self.FF.Hminus(w)
        Hminus = -self.FF.Hplus(w)
        Hzero = -self.FF.Hzero(w)
        rate_VplusA =  self.f(w) / 3 * self.N0 * self.Vcb ** 2 * (64 / 3. * np.pi * Hminus ** 2 + 64 / 3. * np.pi * Hplus ** 2 + 64 / 3. * np.pi * Hzero ** 2)

        return self.VminusA * rate_VminusA + self.VplusA * rate_VplusA


    def dGamma_dcosL(self, cosL: float) -> float:
        """[summary]

        Nota bene:
            * Not fully generalized (yet), w, cosV and chi are integrated over the full range.

        Recommended use cases:
            * Plotting.

        Args:
            cosL (float): [description]

        Returns:
            float: [description]
        """
        assert self.cosL_min <= cosL <= self.cosL_max
        Hplus = lambda w: self.FF.Hplus(w)
        Hminus = lambda w: self.FF.Hminus(w)
        Hzero = lambda w: self.FF.Hzero(w)
        rate_VminusA = quad(
            lambda w: self.f(w) / 3 * self.N0 * self.Vcb ** 2 * (8 * (1 + cosL) ** 2 * np.pi * Hminus(w) ** 2 + 8 * (-1 + cosL) ** 2 * np.pi * Hplus(w) ** 2 - 16 * (-1 + cosL ** 2) * np.pi * Hzero(w) ** 2),
            self.w_min,
            self.w_max
            )[0]

        Hplus = lambda w: -self.FF.Hminus(w)
        Hminus = lambda w: -self.FF.Hplus(w)
        Hzero = lambda w: -self.FF.Hzero(w)   
        rate_VplusA = quad(
            lambda w: self.f(w) / 3 * self.N0 * self.Vcb ** 2 * (8 * (1 + cosL) ** 2 * np.pi * Hminus(w) ** 2 + 8 * (-1 + cosL) ** 2 * np.pi * Hplus(w) ** 2 - 16 * (-1 + cosL ** 2) * np.pi * Hzero(w) ** 2),
            self.w_min,
            self.w_max
            )[0]

        return self.VminusA * rate_VminusA + self.VplusA * rate_VplusA


    def dGamma_dcosV(self, cosV: float) -> float:
        """[summary]

        Nota bene:
            * Not fully generalized (yet), w, cosL and chi are integrated over the full range.

        Recommended use cases:
            * Plotting.

        Args:
            cosV (float): [description]

        Returns:
            float: [description]
        """
        assert self.cosV_min <= cosV <= self.cosV_max
        Hplus = lambda w: self.FF.Hplus(w)
        Hminus = lambda w: self.FF.Hminus(w)
        Hzero = lambda w: self.FF.Hzero(w)
        rate_VminusA = quad(
            lambda w: self.f(w) / 3 * self.N0 * self.Vcb ** 2 * (-16 * (-1 + cosV ** 2) * np.pi * Hminus(w) ** 2 - 16 * (-1 + cosV ** 2) * np.pi * Hplus(w) ** 2 + 32 * cosV ** 2 * np.pi * Hzero(w) ** 2),
            self.w_min,
            self.w_max
            )[0]

        Hplus = lambda w: -self.FF.Hminus(w)
        Hminus = lambda w: -self.FF.Hplus(w)
        Hzero = lambda w: -self.FF.Hzero(w)         
        rate_VplusA = quad(
            lambda w: self.f(w) / 3 * self.N0 * self.Vcb ** 2 * (-16 * (-1 + cosV ** 2) * np.pi * Hminus(w) ** 2 - 16 * (-1 + cosV ** 2) * np.pi * Hplus(w) ** 2 + 32 * cosV ** 2 * np.pi * Hzero(w) ** 2),
            self.w_min,
            self.w_max
            )[0]

        return self.VminusA * rate_VminusA + self.VplusA * rate_VplusA

    def dGamma_dchi(self, chi: float) -> float:
        """[summary]

        Nota bene:
            * Not fully generalized (yet), w, cosL and cosV are integrated over the full range.

        Recommended use cases:
            * Plotting.

        Args:
            cosL (float): [description]

        Returns:
            float: [description]
        """
        assert self.chi_min <= chi <= self.chi_max
        Hplus = lambda w: self.FF.Hplus(w)
        Hminus = lambda w: self.FF.Hminus(w)
        Hzero = lambda w: self.FF.Hzero(w)
        rate_VminusA =  quad(
            lambda w: self.f(w) / 9 * self.N0 * self.Vcb ** 2 * (32 * Hminus(w) ** 2 - 32 * np.cos(2 * chi) * Hminus(w) * Hplus(w) + 32 * Hplus(w) ** 2 + 32 * Hzero(w) ** 2),
            self.w_min, 
            self.w_max
            )[0]

        Hplus = lambda w: -self.FF.Hminus(w)
        Hminus = lambda w: -self.FF.Hplus(w)
        Hzero = lambda w: -self.FF.Hzero(w)         
        rate_VplusA =  quad(
            lambda w: self.f(w) / 9 * self.N0 * self.Vcb ** 2 * (32 * Hminus(w) ** 2 - 32 * np.cos(2 * chi) * Hminus(w) * Hplus(w) + 32 * Hplus(w) ** 2 + 32 * Hzero(w) ** 2),
            self.w_min, 
            self.w_max
            )[0]

        return self.VminusA * rate_VminusA + self.VplusA * rate_VplusA

    # def dGamma_max(self) -> float:
    #     """Return the maximum of the rate.
 
    #     This is used for the generator feature. 
 
    #     Nota bene: If the parameters change (Vxb and/or the form factors, this value will also change).
 
    #     Returns:
    #         float: Maximum of the differential decay rate.
    #     """
    #     return scipy.optimize.fmin(
    #         lambda x: -self.dGamma_dw_dcosL_dcosV_dchi(*x),
    #         np.array([
    #             (self.w_max + self.w_min) / 2,
    #             (self.cosL_max + self.cosL_min) / 2,
    #             (self.cosV_max + self.cosV_min) / 2,
    #             (self.chi_max + self.chi_min) / 2
    #             ]),
    #         disp=False
    #         )


    # def sample_points(self, N) -> np.array:
    #     """Use the hit-or-miss method until a single point is found.
 
    #     Args:
    #         N (int): Sampling size to make efficient use of the random number generator.
 
    #     Returns:
    #         list: Returns a set of random points (w, cosL, cosV, chi). The length of the array is non-deterministic (generator efficiency * sampling size).
    #     """
    #     x = np.array([
    #         scipy.stats.uniform.rvs(self.w_min, self.w_max - self.w_min, size=N),
    #         scipy.stats.uniform.rvs(self.cosL_min, self.cosL_max - self.cosL_min, size=N),
    #         scipy.stats.uniform.rvs(self.cosV_min, self.cosV_max - self.cosV_min, size=N),
    #         scipy.stats.uniform.rvs(self.chi_min, self.chi_max - self.chi_min, size=N)
    #     ]).transpose()
 
    #     f = scipy.stats.uniform.rvs(0, self.rate_max, size=N)
    #     return [(*_x, _f) for _x, _f in zip(x, f) if self.dGamma_dw_dcosL_dcosV_dchi(*_x) > _f]
 
 
    # def generate_events(self, N):
    #     """Generate the requested number of events.
 
    #     The efficiency of the hit-or.miss method is roughly 25%. To make efficient use of the random number generator of scipy, we over-sample by a factor of 5.
    #     This way we should usually be able to generate the requested number of events in one go.
 
    #     Args:
    #         N (int): Number of events to be generated.
 
    #     Returns:
    #         np.array: Array of random data points (w, cosL, cosV, chi) drawn from the differentical decay rate.
    #     """
    #     events = []
    #     while len(events) < N:
    #         events += self.sample_points(N*5)
    #     return np.array(events[:N])


if __name__ == "__main__":
    pass
