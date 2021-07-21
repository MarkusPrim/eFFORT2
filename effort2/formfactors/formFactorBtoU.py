import abc

from effort2.formfactors.formFactorBase import FormFactor


class FormFactorBToU(FormFactor):
    r"""This class defines the interface for any form factor parametrization to be used in conjunction with the rate implementations
for $B \to P \ell \nu_\ell$ and $B \to V \ell \nu_\ell$ decays, where P stands for Pseudoscalar and V stands for Vector light mesons.
    """

    def __init__(
        self, 
        m_B: float, 
        m_M: float, 
        m_L: float = 0
        ) -> None:
        r"""[summary]

        Args:
            m_B (float): Mass of the B meson.
            m_M (float): Mass of the final state meson.
            m_L (float): Mass of the final state lepton. Defaults to 0 (zero lepton mass approximation).
        """
        super().__init__(m_B, m_M, m_L)


    def kaellen(self, q2):
        return ((self.m_B + self.m_M) ** 2 - q2) * ((self.m_B - self.m_M) ** 2 - q2)


    def Hplus(self, w: float) -> float:
        q2 = self.q2(w)
        return self.kaellen(q2) ** 0.5 * self.V(q2) / (self.m_B + self.m_M) + (self.m_B + self.m_M) * self.A1(q2)


    def Hminus(self, w: float) -> float:
        q2 = self.q2(w)
        return self.kaellen(q2) ** 0.5 * self.V(q2) / (self.m_B + self.m_M) - (self.m_B + self.m_M) * self.A1(q2)


    def Hzero(self, w: float) -> float:
        q2 = self.q2(w)
        return 8 * self.m_B * self.m_M / q2 ** 0.5 * self.A12(q2)


    def Hscalar(self, w: float) -> float:
        q2 = self.q2(w)
        return self.kaellen(q2) ** 0.5 / q2 ** 0.5 * self.A0(q2)


    @abc.abstractmethod
    def A0(self, q2: float) -> float:
        pass


    @abc.abstractmethod
    def A1(self, q2: float) -> float:
        pass


    @abc.abstractmethod
    def A12(self, q2: float) -> float:
        pass


    @abc.abstractmethod
    def V(self, q2: float) -> float:
        pass


class BToRhoBSZ(FormFactorBToU):

    def __init__(
        self,
        m_B: float,
        m_M: float, 
        m_L: float,
        A0_i: tuple,
        A1_i: tuple,
        A12_i: tuple,
        V_i: tuple,
        T1_i: tuple,
        T2_i: tuple,
        T23_i: tuple,
        lambdaBar: float = 0.5,
        pole_masses: dict = None,
        ) -> None:
        super().__init__(m_B, m_M, m_L)
        self.m_b = m_B - lambdaBar
        self.m_u = 0
        self.set_expansion_coefficients(A0_i, A1_i, A12_i, V_i, T1_i, T2_i, T23_i)
        if pole_masses is None:
            self.pole_masses = {
                "A0": 5.279,
                "A1": 5.724,
                "A12": 5.724,
                "V": 5.325,
                "T1": 5.325,
                "T2": 5.724,
                "T23": 5.724,
            }
        else:
            self.pole_masses = pole_masses
        self.tplus = (self.m_B + self.m_M) ** 2
        self.tminus = (self.m_B - self.m_M) ** 2
        self.tzero = self.tplus * (1 - (1 - self.tminus / self.tplus) ** 0.5)
        

    def set_expansion_coefficients(
        self,
        A0_i: tuple,
        A1_i: tuple,
        A12_i: tuple,
        V_i: tuple,
        T1_i: tuple = (),
        T2_i: tuple = (),
        T23_i: tuple = (),
        ) -> None:
        """Sets the expansion coefficients and imposes the constraint on alpha_A0_0 and alpha_T2_0.

        Expects the coefficients in the following form:
            * A0_1, A0_2, ...
            * A1_0, A1_1, A1_2, ...
            * A12_0, A12_1, A12_2, ...
            * V_0, V_1, V_2, ...
            * T1_0, T1, T2, ...
            * T2_1, T2_2, ...
            * T23_0, T23_1, T23_2, ...
        and automaticalle imposes the constraint on A0_0 and T2_0.. 
        The order for the expansion can be chosen arbitrarily.

        Args:
            A0_i (tuple): Expansion coefficients for the form factor A0.
            A1_i (tuple): Expansion coefficients for the form factor A1.
            A12_i (tuple): Expansion coefficients for the form factor A12.
            V_i (tuple): Expansion coefficients for the form factor V.
            T1_i (tuple): Expansion coefficients for the form factor T1. Not required for the SM calculations here.
            T2_i (tuple): Expansion coefficients for the form factor T2. Not required for the SM calculations here.
            T23_i (tuple): Expansion coefficients for the form factor T23. Not required for the SM calculations here.
        """
        self.expansion_coefficients_A0 = [8 * self.m_B * self.m_M / (self.m_B ** 2 - self.m_M ** 2) * A12_i[0], *A0_i]
        self.expansion_coefficients_A1 = [*A1_i]
        self.expansion_coefficients_A12 = [*A12_i]
        self.expansion_coefficients_V = [*V_i]
        self.expansion_coefficients_T1 = [*T1_i]
        try:
            self.expansion_coefficients_T2 = [T1_i[0], *T2_i]
        except IndexError:
            self.expansion_coefficients_T2 = []
        self.expansion_coefficients_T23 = [*T23_i]


    def z(self, q2):
        return ((self.tplus - q2) ** 0.5 - (self.tplus - self.tzero) ** 0.5) / (
                (self.tplus - q2) ** 0.5 + (self.tplus - self.tzero) ** 0.5)


    def blaschke_pole(self, q2, m_pole):
        return (1 - q2 / m_pole ** 2) ** -1


    def form_factor(self, q2, m_pole, coefficients):
        return self.blaschke_pole(q2, m_pole) * sum(
            [par * (self.z(q2) - self.z(0)) ** k for k, par in enumerate(coefficients)])


    def AP(self, q2):
        return -2 * self.m_M / (self.m_b + self.m_u) * self.A0(q2)


    def A0(self, q2):
        return self.form_factor(q2, self.pole_masses["A0"], self.expansion_coefficients_A0)


    def A1(self, q2):
        return self.form_factor(q2, self.pole_masses["A1"], self.expansion_coefficients_A1)


    def A12(self, q2):
        return self.form_factor(q2, self.pole_masses["A12"], self.expansion_coefficients_A12)


    def A2(self, q2):
        return ((self.m_B + self.m_M) ** 2 * (self.m_B ** 2 - self.m_M ** 2 - q2) * self.A1(w)
            - 16 * self.m_B * self.m_M ** 2 * (self.m_B + self.m_M) * self.A12(w)) / (2 * self.m_B * self.m_M) ** 2


    def V(self, q2):
        return self.form_factor(q2, self.pole_masses["V"], self.expansion_coefficients_V)


    def T1(self, q2):
        return self.form_factor(q2, self.pole_masses["T1"], self.expansion_coefficients_T1)


    def T2(self, q2):
        return self.form_factor(q2, self.pole_masses["T2"], self.expansion_coefficients_T2)


    def T23(self, q2):
        return self.form_factor(q2, self.pole_masses["T23"], self.expansion_coefficients_T23)


BToOmegaBSZ = BToRhoBSZ
