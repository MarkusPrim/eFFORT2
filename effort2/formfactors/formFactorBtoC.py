import abc
import numpy as np

from effort2.formfactors.formFactorBase import FormFactor


class FormFactorBToC(FormFactor):
    r"""This class defines the interface for any form factor parametrization to be used in conjunction with the rate implementations
for $B \to P \ell \nu_\ell$ and $B \to V \ell \nu_\ell$ decays, where P stands for Pseudoscalar and V stands for Vector heavy mesons.
    """

    def __init__(self, m_B, m_M) -> None:
        r"""[summary]

        Args:
            m_B (float): Mass of the B meson.
            m_M (float): Mass of the final state meson.
        """
        super().__init__(m_B, m_M)
        self.rprime = 2 * np.sqrt(self.m_B * self.m_M) / (self.m_B + self.m_M)


    def A0(self, w: float) -> float:
        raise RuntimeError("Not implemented. But also not required for light leptons.")


    def A1(self, w: float) -> float:
        return (w + 1) / 2 * self.rprime * self.h_A1(w)


    def A2(self, w: float) -> float:
        return self.R2(w) / self.rprime * self.h_A1(w)


    def V(self, w: float) -> float:
        return self.R1(w) / self.rprime * self.h_A1(w)


    def Hplus(self, w: float) -> float:
        return (self.m_B + self.m_M) * self.A1(w) - 2 * self.m_B / (self.m_B + self.m_M) * self.m_M * (
                w ** 2 - 1) ** 0.5 * self.V(w)


    def Hminus(self, w: float) -> float:
        return (self.m_B + self.m_M) * self.A1(w) + 2 * self.m_B / (self.m_B + self.m_M) * self.m_M * (
                w ** 2 - 1) ** 0.5 * self.V(w)


    def Hzero(self, w: float) -> float:
        m_B = self.m_B
        m_M = self.m_M
        q2 = (m_B ** 2 + m_M ** 2 - 2 * w * m_B * m_M)
        return 1 / (2 * m_M * q2 ** 0.5) * ((m_B ** 2 - m_M ** 2 - q2) * (m_B + m_M) * self.A1(w)
                                            - 4 * m_B ** 2 * m_M ** 2 * (w ** 2 - 1) / (m_B + m_M) * self.A2(w))


    def Hscalar(self) -> None:
        raise RuntimeError("Not implemented. But also not required for light leptons.")


    @abc.abstractmethod
    def h_A1(self, w: float) -> float:
        pass


    @abc.abstractmethod
    def R0(self, w: float) -> float:
        raise RuntimeError("Not implemented. But also not required for light leptons.")


    @abc.abstractmethod
    def R1(self, w: float) -> float:
        pass


    @abc.abstractmethod
    def R2(self, w: float) -> float:
        pass


    def z(self, w: float) -> float:
        """Variable for the expansion used in BGL and CLN.

        Args:
            w (float): [description]

        Returns:
            float: [description]
        """
        term1 = np.sqrt(w + 1)
        term2 = np.sqrt(2)
        return (term1 - term2) / (term1 + term2)


class BToDStarCLN(FormFactorBToC):

    def __init__(
        self,
        m_B: float,
        m_M: float,
        h_A1_1: float,
        rho2: float, 
        R1_1: float, 
        R2_1: float,
        ):
        """[summary]

        Args:
            m_B (float): [description]
            m_M (float): [description]
            h_A1_1 (float, optional): [description]
            rho2 (float, optional): [description]
            R1_1 (float, optional): [description]
            R2_1 (float, optional): [description]
        """
        super().__init__(m_B, m_M)
        self.h_A1_1 = h_A1_1
        self.rho2 = rho2
        self.R1_1 = R1_1
        self.R2_1 = R2_1


    def h_A1(self, w: float) -> float:
        """[summary]

        Args:
            w (float): [description]

        Returns:
            float: [description]
        """
        rho2 = self.rho2
        z = self.z(w)
        return self.h_A1_1 * (1 - 8 * rho2 * z + (53 * rho2 - 15) * z ** 2 - (231 * rho2 - 91) * z ** 3)


    def R0(self) -> None:
        raise RuntimeError("Not implemented. But also not required for light leptons.")


    def R1(self, w: float) -> float:
        """[summary]

        Args:
            w (float): [description]

        Returns:
            float: [description]
        """
        return self.R1_1 - 0.12 * (w - 1) + 0.05 * (w - 1) ** 2


    def R2(self, w: float) -> float:
        """[summary]

        Args:
            w (float): [description]

        Returns:
            float: [description]
        """
        return self.R2_1 + 0.11 * (w - 1) - 0.06 * (w - 1) ** 2


class BToDStarBGL(FormFactorBToC):

    def __init__(
        self, 
        m_B: float, 
        m_M: float, 
        exp_coeff_a: tuple,
        exp_coeff_b: tuple,
        exp_coeff_c: tuple,
        chiT_plus33: float = 5.28e-4,
        chiT_minus33: float = 3.07e-4,
        n_i: float = 2.6,
        axialvector_poles: list = [6.730, 6.736, 7.135, 7.142],
        vector_poles: list = [6.337, 6.899, 7.012, 7.280], 
        ) -> None:
        super().__init__(m_B, m_M)

        # BGL specifics, default is given in arXiv:1703.08170v2
        self.chiT_plus33 = chiT_plus33
        self.chiT_minus33 = chiT_minus33
        self.n_i = n_i # effective number of light quarks
        self.axialvector_poles = axialvector_poles
        self.vector_poles = vector_poles
        self.r = m_M / m_B
        self.set_expansion_coefficients(exp_coeff_a, exp_coeff_b, exp_coeff_c)


    def set_expansion_coefficients(self, exp_coeff_a: tuple, exp_coeff_b: tuple, exp_coeff_c: tuple) -> None:
        """Sets the expansion coefficients and imposes the constraint on c0.

        Expects the coefficients in the following form:
            * a0, a1, ...
            * b0, b1, ...
            * c1, c2 ...
        and automaticalle imposes the constraint on c0. 
        The order for the coefficients can be chosen arbitrarily.

        Args:
            exp_coeff_a ([tuple]): Expansion coefficients for the form factor g.
            exp_coeff_b ([tuple]): Expansion coefficients for the form factor f.
            exp_coeff_c ([tuple]): Expansion coefficients for the form factor F1
        """
        self.expansion_coefficients_a = [*exp_coeff_a]
        self.expansion_coefficients_b = [*exp_coeff_b]
        self.expansion_coefficients_c = [((self.m_B - self.m_M) * self.phi_F1(0) / self.phi_f(0)) * exp_coeff_b[0], *exp_coeff_c]


    def h_A1(self, w):
        z = self.z(w)
        return self.f(z) / (self.m_B * self.m_M) ** 0.5 / (1 + w)


    def R0(self) -> None:
        raise RuntimeError("Not implemented. But also not required for light leptons.")


    def R1(self, w):
        z = self.z(w)
        return (w + 1) * self.m_B * self.m_M * self.g(z) / self.f(z)


    def R2(self, w):
        z = self.z(w)
        return (w - self.r) / (w - 1) - self.F1(z) / self.m_B / (w - 1) / self.f(z)


    def g(self, z):
        return self.BGL_form_factor(z, lambda x: self.blaschke_factor(x, self.vector_poles), self.phi_g,
                               self.expansion_coefficients_a)


    def f(self, z):
        return self.BGL_form_factor(z, lambda x: self.blaschke_factor(x, self.axialvector_poles), self.phi_f,
                               self.expansion_coefficients_b)


    def F1(self, z):
        return self.BGL_form_factor(z, lambda x: self.blaschke_factor(x, self.axialvector_poles), self.phi_F1,
                               self.expansion_coefficients_c)


    def blaschke_factor(self, z, poles):
        return np.multiply.reduce([(z - self.z_p(m_pole)) / (1 - z * self.z_p(m_pole)) for m_pole in poles])


    def z_p(self, m_pole):
        m_B = self.m_B
        m_M = self.m_M
        term1 = ((m_B + m_M) ** 2 - m_pole ** 2) ** 0.5
        term2 = ((m_B + m_M) ** 2 - (m_B - m_M) ** 2) ** 0.5
        return (term1 - term2) / (term1 + term2)
 

    def phi_g(self, z):
        r = self.r
        return (256 * self.n_i / 3 / np.pi / self.chiT_plus33) ** 0.5 \
                * r ** 2 * (1 + z) ** 2 * (1 - z) ** -0.5 / ((1 + r) * (1 - z) + 2 * r ** 0.5 * (1 + z)) ** 4
 

    def phi_f(self, z):
        r = self.r
        return 1 / self.m_B ** 2 * (16 * self.n_i / 3 / np.pi / self.chiT_minus33) ** 0.5 \
                * r * (1 + z) * (1 - z) ** (3. / 2) / ((1 + r) * (1 - z) + 2 * r ** 0.5 * (1 + z)) ** 4
 

    def phi_F1(self, z):
        r = self.r
        return 1 / self.m_B ** 3 * (8 * self.n_i / 3 / np.pi / self.chiT_minus33) ** 0.5 \
               * r * (1 + z) * (1 - z) ** (5. / 2) / ((1 + r) * (1 - z) + 2 * r ** 0.5 * (1 + z)) ** 5


    def BGL_form_factor(self, z: float, p: float, phi: float, a: list) -> float:
        """Calculates the BGL form factor.

        Args:
            z (float): BGL expansion parameter.
            p (float): Corresponding Blaschke factors.
            phi (float): Corresponding outer function.
            a (list): Corresponding expansion coefficients.

        Returns:
            float: BGL form factor at given z.
        """

        return 1 / (p(z) * phi(z)) * sum([a_i * z ** n for n, a_i in enumerate(a)])
