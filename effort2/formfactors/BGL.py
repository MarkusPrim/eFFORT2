import functools
import numba as nb
import numpy as np
from effort2.formfactors.HelicityAmplitudes import FormFactorHQETBToP, FormFactorHQETBToV


def BGL_form_factor(z: float, p: float, phi: float, a: list) -> float:
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



class BToDBGL(FormFactorHQETBToP):
    """_summary_

    Reference: https://arxiv.org/pdf/1503.07237.pdf

    Args:
        FormFactorHQETBToP (_type_): _description_
    """

    def __init__(
            self, 
            m_B: float, 
            m_P: float, 
            exp_coeff_plus: tuple,
            exp_coeff_zero: tuple,
            m_L: float = 0
            ) -> None:
        super().__init__(m_B, m_P, m_L)
        self.BGL_form_factor = BGL_form_factor
        self.set_expansion_coefficients(exp_coeff_plus, exp_coeff_zero)


    def set_expansion_coefficients(self, exp_coeff_plus: tuple, exp_coeff_zero: tuple) -> None:
        """Sets the expansion coefficients and imposes the constraint on c0.

        Expects the coefficients in the following form:
            * a+0, a+1, ...
            * a00, a01, ...
        The order for the expansion can be chosen arbitrarily.

        Args:
            exp_coeff_plus ([tuple]): Expansion coefficients for the form factor fplus.
            exp_coeff_zero ([tuple]): Expansion coefficients for the form factor fzero.
        """
        self.expansion_coefficients_plus = [*exp_coeff_plus]
        self.expansion_coefficients_zero = [*exp_coeff_zero]


    def fplus(self, w: float) -> float:
        z = self.z(w)
        return self.BGL_form_factor(z, lambda x: self.blaschke_factor_plus(x), self.phi_plus, self.expansion_coefficients_plus)


    def fzero(self, w:float) -> float:
        z = self.z(w)
        return self.BGL_form_factor(z, lambda x: self.blaschke_factor_zero(x), self.phi_zero, self.expansion_coefficients_zero)

    
    def blaschke_factor_plus(self, z: float) -> float:
        return 1
    

    def blaschke_factor_zero(self, z: float) -> float:
        return 1
    

    def phi_plus(self, z: float) -> float:
        r = self.r
        return 1.1213 * (1 + z) ** 2 * (1 - z) ** (1 / 2) * ((1 + r) * (1 - z) + 2 * r ** 0.5 * (1 + z)) ** -5
    
    
    def phi_zero(self, z: float) -> float:
        r = self.r
        return 0.5299 * (1 + z) * (1 - z) ** (3 / 2) * ((1 + r) * (1 - z) + 2 * r ** 0.5 * (1 + z)) ** -4


    def __str__(self):
        return f"""BGL B --> D Expansion coefficients
a+ = {self.expansion_coefficients_plus}
a0 = {self.expansion_coefficients_zero}"""


class BToDStarBGL(FormFactorHQETBToV):
    """_summary_

    Reference: https://arxiv.org/pdf/2105.14019.pdf
    TODO: Revalidate the implementation against this specific reference.

    Args:
        FormFactorHQETBToV (_type_): _description_
    """

    def __init__(
        self, 
        m_B: float, 
        m_V: float, 
        exp_coeff_a: tuple,
        exp_coeff_b: tuple,
        exp_coeff_c: tuple,
        exp_coeff_d: tuple = (0, ),
        m_L: float = 0,
        chiT_plus33: float = 5.28e-4,  # TODO: Update?
        chiT_minus33: float = 3.07e-4, # TODO: Update?
        chiL_1plus: float = 1.9421e-2, # TODO: Check
        n_i: float = 2.6,
        axialvector_poles: list = [6.730, 6.736, 7.135, 7.142],   # TODO: Update?
        vector_poles: list = [6.337, 6.899, 7.012, 7.280],        # TODO: Update?
        pseudoscalar_poles: list = [6.275, 6.842, 7.250],         # TODO: Check
        ) -> None:
        super().__init__(m_B, m_V, m_L)
        self.BGL_form_factor = BGL_form_factor

        # BGL specifics, default is given in arXiv:1703.08170v2
        self.chiT_plus33 = chiT_plus33
        self.chiT_minus33 = chiT_minus33
        self.chiL_1plus = chiL_1plus  # TODO: Not consistent with the chiT values, this one is from lattice paper
        self.n_i = n_i # effective number of light quarks
        self.axialvector_poles = axialvector_poles
        self.vector_poles = vector_poles
        self.pseudoscalar_poles = pseudoscalar_poles  # TODO: Not consistent with the chiT values, this one is from lattice paper
        self.r = m_V / m_B
        self.set_expansion_coefficients(exp_coeff_a, exp_coeff_b, exp_coeff_c, exp_coeff_d=exp_coeff_d)


    def set_expansion_coefficients(self, exp_coeff_a: tuple, exp_coeff_b: tuple, exp_coeff_c: tuple, exp_coeff_d: tuple = (0, )) -> None:
        """Sets the expansion coefficients and imposes the constraint on c0.

        Expects the coefficients in the following form:
            * a0, a1, ...
            * b0, b1, ...
            * c1, c2 ...
            * d0, c1 ...
        and automaticalle imposes the constraint on c0. 
        The order for the expansion can be chosen arbitrarily.

        Args:
            exp_coeff_a ([tuple]): Expansion coefficients for the form factor g.
            exp_coeff_b ([tuple]): Expansion coefficients for the form factor f.
            exp_coeff_c ([tuple]): Expansion coefficients for the form factor F1
            exp_coeff_d ([tuple]): Expansion coefficients for the form factor F2
        """
        self.expansion_coefficients_a = [*exp_coeff_a]
        self.expansion_coefficients_b = [*exp_coeff_b]
        self.expansion_coefficients_c = [((self.m_B - self.m_V) * self.phi_F1(0) / self.phi_f(0)) * exp_coeff_b[0], *exp_coeff_c]
        self.expansion_coefficients_d = [*exp_coeff_d]


    def Hplus(self, w: float) -> float:
        """Overwritten generic function from parent class, because this is numerical more stable when fitting."""
        m_B = self.m_B
        z = self.z(w)
        q2 = self.kinematics.q2(w)
        return self.f(z) - m_B * self.kinematics.p(q2) * self.g(z)


    def Hminus(self, w: float) -> float:
        """Overwritten generic function from parent class, because this is numerical more stable when fitting."""
        m_B = self.m_B
        z = self.z(w)
        q2 = self.kinematics.q2(w)
        return self.f(z) + m_B * self.kinematics.p(q2) * self.g(z) 


    def Hzero(self, w: float) -> float:
        """Overwritten generic function from parent class, because this is numerical more stable when fitting."""
        z = self.z(w)
        q2 = self.kinematics.q2(w)
        return self.F1(z) / q2 ** 0.5


    def h_A1(self, w):
        z = self.z(w)
        return self.f(z) / (self.m_B * self.m_V) ** 0.5 / (1 + w)

    
    def h_A2(self, w):
        z = self.z(w)
        f = self.f(z)
        F1 = self.F1(z)
        F2 = self.F2(z)
        mB = self.m_B
        r = self.r
        return -((
            f * mB + F1 * r - F2 * mB ** 2 * r + f * mB * r ** 2 - F1 * w - 2 * f * mB * r * w + F2 * mB ** 2 * r * w ** 2
            ) / (
        mB ** 2 * r ** 0.5 * (-1 + w) * (1 + w) * (1 + r ** 2 - 2 * r * w)
        ))


    # def h_A2(self, w):
    #     r = self.r
    #     hA1 = self.h_A1(w)
    #     R0 = self.R0(w)
    #     R2 = self.R2(w)
    #     return -hA1 * (-1 + R0 + r * R0 - r * R2 - w + R2 * w) / (1 + r ** 2 - 2 * r * w)


    def h_A3(self, w):
        z = self.z(w)
        f = self.f(z)
        F1 = self.F1(z)
        F2 = self.F2(z)
        mB = self.m_B
        r = self.r
        return -((
            F1 + F2 * mB ** 2 * r ** 2 - f * mB * w - F1 * r * w - f * mB * r ** 2 * w + 2 * f * mB * r * w ** 2 - F2 * mB ** 2 * r ** 2 * w ** 2
            ) / (
        mB ** 2 * r ** 0.5 * (-1 + w) * (1 + w) * (1 + r ** 2 - 2 * r * w)
        ))


    # def h_A3(self, w):
    #     r = self.r
    #     hA1 = self.h_A1(w)
    #     R0 = self.R0(w)
    #     R2 = self.R2(w)
    #     return -(hA1 * r - hA1 + r * R0 - hA1 * r ** 2 * R0 - hA1 * R2 + hA1 * r * w + hA1 * r * R2 * w) / (1 + r ** 2 - 2 * r * w)


    def h_V(self, w):
        z = self.z(w)
        return self.r ** 0.5 * self.m_B * self.g(z)


    # def h_V(self, w):
    #     hA1 = self.h_A1(w)
    #     R1 = self.R1(w)
    #     return hA1 * R1


    def R0(self, w):
        z = self.z(w)
        return self.r ** 0.5 / (1 + self.r) * self.F2(z) / (self.f(z) / (self.m_B * self.m_V) ** 0.5 / (1 + w))


    def R1(self, w):
        z = self.z(w)
        return (w + 1) * self.m_B * self.m_V * self.g(z) / self.f(z)


    def R2(self, w):
        z = self.z(w)
        return (w - self.r) / (w - 1) - self.F1(z) / self.m_B / (w - 1) / self.f(z)


    def g(self, z):
        return self.BGL_form_factor(z, lambda x: self.blaschke_factor_vector(x), self.phi_g,
                               self.expansion_coefficients_a)


    def f(self, z):
        return self.BGL_form_factor(z, lambda x: self.blaschke_factor_axialvector(x), self.phi_f,
                               self.expansion_coefficients_b)


    def F1(self, z):
        return self.BGL_form_factor(z, lambda x: self.blaschke_factor_axialvector(x), self.phi_F1,
                               self.expansion_coefficients_c)

    
    def F2(self, z):
        return self.BGL_form_factor(z, lambda x: self.blaschke_factor_pseudoscalar(x), self.phi_F2,
                        self.expansion_coefficients_d)


    @functools.lru_cache()
    def blaschke_factor_vector(self, z):
        return np.multiply.reduce([(z - self.z_p(m_pole)) / (1 - z * self.z_p(m_pole)) for m_pole in self.vector_poles])

    
    @functools.lru_cache()
    def blaschke_factor_axialvector(self, z):
        return np.multiply.reduce([(z - self.z_p(m_pole)) / (1 - z * self.z_p(m_pole)) for m_pole in self.axialvector_poles])


    @functools.lru_cache()
    def blaschke_factor_pseudoscalar(self, z):
        return np.multiply.reduce([(z - self.z_p(m_pole)) / (1 - z * self.z_p(m_pole)) for m_pole in self.pseudoscalar_poles])


    @functools.lru_cache()
    def z_p(self, m_pole):
        return self._z_p(m_pole, self.m_B, self.m_V)


    @staticmethod
    @nb.njit(cache=True)
    def _z_p(m_pole, m_B, m_M):
        term1 = ((m_B + m_M) ** 2 - m_pole ** 2) ** 0.5
        term2 = ((m_B + m_M) ** 2 - (m_B - m_M) ** 2) ** 0.5
        return (term1 - term2) / (term1 + term2)
 

    @functools.lru_cache()
    def phi_g(self, z):
        return self._phi_g(z, self.r, self.n_i, self.chiT_plus33)        
 

    @staticmethod
    @nb.jit(cache=True)
    def _phi_g(z, r, n_i, chiT_plus33):
        return (256 * n_i / 3 / np.pi / chiT_plus33) ** 0.5 \
                * r ** 2 * (1 + z) ** 2 * (1 - z) ** -0.5 / ((1 + r) * (1 - z) + 2 * r ** 0.5 * (1 + z)) ** 4


    @functools.lru_cache()
    def phi_f(self, z):
        return self._phi_f(z, self.r, self.n_i, self.chiT_minus33, self.m_B)
 

    @staticmethod
    @nb.jit(cache=True)
    def _phi_f(z, r, n_i, chiT_minus33, m_B):
        return 1 / m_B ** 2 * (16 * n_i / 3 / np.pi / chiT_minus33) ** 0.5 \
                * r * (1 + z) * (1 - z) ** (3. / 2) / ((1 + r) * (1 - z) + 2 * r ** 0.5 * (1 + z)) ** 4


    @functools.lru_cache()
    def phi_F1(self, z):
        return self._phi_F1(z, self.r, self.n_i, self.chiT_minus33, self.m_B)


    @staticmethod
    @nb.jit(cache=True)
    def _phi_F1(z, r, n_i, chiT_minus33, m_B):
        return 1 / m_B ** 3 * (8 * n_i / 3 / np.pi / chiT_minus33) ** 0.5 \
               * r * (1 + z) * (1 - z) ** (5. / 2) / ((1 + r) * (1 - z) + 2 * r ** 0.5 * (1 + z)) ** 5

    @functools.lru_cache()
    def phi_F2(self, z):
        return self._phi_F2(z, self.r, self.n_i, self.chiL_1plus, self.m_B)


    @staticmethod
    @nb.jit(cache=True)
    def _phi_F2(z, r, n_i, chiL_1plus, m_B):
        return 8 * 2 ** 0.5 * r ** 2 * (n_i / np.pi / chiL_1plus) ** 0.5 \
               * (1 + z) ** 2 * (1 - z) ** -0.5 / ((1 + r) * (1 - z) + 2 * r ** 0.5 * (1 + z)) ** 4


    def __str__(self):
        return f"""BGL B --> D* Expansion coefficients
a = {self.expansion_coefficients_a}
b = {self.expansion_coefficients_b}
c = {self.expansion_coefficients_c}
d = {self.expansion_coefficients_d}"""


if __name__ == "__main__":
    pass