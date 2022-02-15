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
        m_L: float = None,
        G_F: float = 1.1663787e-5,
        eta_EW: float = 1.0066,
        BR_Dstar_decay: float = 1
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
            G_F (float): Effective coupling constant of the weak interaction (Fermi's constant) in units of GeV^-2. Default value from: https://pdg.lbl.gov/2020/reviews/rpp2020-rev-phys-constants.pdf.
            eta_EW (float): Electroweak corrections.
            BR_Dstar_decay (float, optional): In case the D* meson decay is not treated fully inclusive (BR < 1). It is assumed that this value will never change when handling caches.
        """
        assert 0 <= BR_Dstar_decay <= 1
        self.FF = FF

        self.Vcb = Vcb
        self.mB = FF.m_B if m_B is None else m_B
        self.mV = FF.m_Ds if m_V is None else m_V
        self.mL = FF.m_L if m_L is None else m_L
        self.GF = G_F
        self.eta_EW = eta_EW
        self.BR_Dstar_decay = BR_Dstar_decay

        # Boundaries of the 1D rate. These assumptions are imposed in the analytical integrations in Mathematica.
        self.kinematics = Kinematics(self.mB, self.mV, self.mL)
        self.w_min, self.w_max = self.kinematics.w_range


    def dGamma_dw(
        self, 
        w: float,
        ) -> float:
        """Full 4D differential decay rate for B to V(ector) meson decays.

        Args:
            w (float): Recoil against the hadronic system.
        
        Returns:
            float: Rate at the requested phase space point.
        """
        if not self.w_min < w <= self.w_max: return 0

        q2 = self.kinematics.q2(w)
        p = self.kinematics.p(q2)

        return 2 * self.mB * self.mV * self.Vcb ** 2 * self.GF ** 2 / (96*np.pi**3 * self.mB**2) * (1 - self.mL**2/q2)**2 * p * q2 * ( 
            (self.FF.Hplus(w)**2 + self.FF.Hminus(w)**2 + self.FF.Hzero(w)**2)*( 1 + self.mL**2/(2*q2) ) +  3*self.mL**2/(2*q2)*self.FF.Hscalar(w)**2)


    def DGamma_Dw(
        self,
        wmin: float = None,
        wmax: float = None,
        debug: bool = False
        ) -> float:
        """The differential decay rate for B to V(ector) meson decays, where the angular variables are analytically integrated.
        
        The integration over w is performed numerically, because analyitcal integration would introduce a dependency on the chosen form factor parametrization.

        Recommended use case:
            * Fitting. The class also provides interfaces to the marginalized distributions directly.

        Args:
            wmin (float): Recoil against the hadronic system lower boundary.
            wmax (float): Recoil against the hadronic system upper boundary.
            debug (bool): Give integration output.

        Returns:
            float: Rate in the marginalized region of the phase space.
        """
        wmin = self.w_min if wmin is None else wmin
        wmax = self.w_max if wmax is None else wmax

        output = quad(lambda w: self.dGamma_dw(w), wmin, wmax, full_output=debug)

        if not debug:
            return output[0]
        else:
            return output


    def Gamma(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.DGamma_Dw(self.w_min, self.w_max)
