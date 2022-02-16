from effort2.formfactors.formFactorBtoC import FormFactorBToDstar


class BToDStarBGL(FormFactorBToDstar):

    def __init__(
        self, 
        m_B: float, 
        m_V: float, 
        exp_coeff_a: tuple,
        exp_coeff_b: tuple,
        exp_coeff_c: tuple,
        chiT_plus33: float = 5.28e-4,
        chiT_minus33: float = 3.07e-4,
        n_i: float = 2.6,
        axialvector_poles: list = [6.730, 6.736, 7.135, 7.142],
        vector_poles: list = [6.337, 6.899, 7.012, 7.280], 
        ) -> None:
        super().__init__(m_B, m_V)

        # BGL specifics, default is given in arXiv:1703.08170v2
        self.chiT_plus33 = chiT_plus33
        self.chiT_minus33 = chiT_minus33
        self.n_i = n_i # effective number of light quarks
        self.axialvector_poles = axialvector_poles
        self.vector_poles = vector_poles
        self.r = m_V / m_B
        self.set_expansion_coefficients(exp_coeff_a, exp_coeff_b, exp_coeff_c)


    def set_expansion_coefficients(self, exp_coeff_a: tuple, exp_coeff_b: tuple, exp_coeff_c: tuple) -> None:
        """Sets the expansion coefficients and imposes the constraint on c0.

        Expects the coefficients in the following form:
            * a0, a1, ...
            * b0, b1, ...
            * c1, c2 ...
        and automaticalle imposes the constraint on c0. 
        The order for the expansion can be chosen arbitrarily.

        Args:
            exp_coeff_a ([tuple]): Expansion coefficients for the form factor g.
            exp_coeff_b ([tuple]): Expansion coefficients for the form factor f.
            exp_coeff_c ([tuple]): Expansion coefficients for the form factor F1
        """
        self.expansion_coefficients_a = [*exp_coeff_a]
        self.expansion_coefficients_b = [*exp_coeff_b]
        self.expansion_coefficients_c = [((self.m_B - self.m_V) * self.phi_F1(0) / self.phi_f(0)) * exp_coeff_b[0], *exp_coeff_c]


    def h_A1(self, w):
        z = self.z(w)
        return self.f(z) / (self.m_B * self.m_V) ** 0.5 / (1 + w)


    def R0(self) -> None:
        raise RuntimeError("Not implemented. But also not required for light leptons.")


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


    @functools.lru_cache()
    def blaschke_factor_vector(self, z):
        return np.multiply.reduce([(z - self.z_p(m_pole)) / (1 - z * self.z_p(m_pole)) for m_pole in self.vector_poles])

    
    @functools.lru_cache()
    def blaschke_factor_axialvector(self, z):
        return np.multiply.reduce([(z - self.z_p(m_pole)) / (1 - z * self.z_p(m_pole)) for m_pole in self.axialvector_poles])


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


if __name__ == "__main__":
    pass