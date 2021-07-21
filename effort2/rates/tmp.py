        def dGamma_dw_DcosL_DcosV_Dchi(
            self,
            w: float,
            cosLmin: float,
            cosLmax: float,
            cosVmin: float,
            cosVmax: float,
            chimin: float,
            chimax: float,
            ) -> float:
            """The differential decay rate for B to V(ector) meson decays, where the angular variables are analytically integrated.
            
            The integration over w is not performed, because this would introduce a dependency on the chosen form factor parametrization.

            Args:
                w (float): Recoil against the hadronic system.
                cosLmin (float): Lepton angle lower boundary.
                cosLmax (float): Lepton angle upper boundary.
                cosVmin (float): Vector meson angle lower boundary.
                cosVmax (float): Vector meson angle upper boundary.
                chimin (float): Decay plane angle lower boundary.
                chimax (float): Decay plane angle upper boundary.

            Returns:
                float: Rate at the requested phase space point w, within the marginalized region of the angular phase space.
            """
            assert self.w_min <= w <= self.w_max
            assert self.cosL_min <= cosLmin < cosLmax <= self.cosL_max
            assert self.cosL_min <= cosLmin < cosLmax <= self.cosL_max
            assert self.chi_min <= chimin < chimax <= self.chi_max

            fw = (1 - 2 * w * self.r + self.r ** 2) * (w ** 2 - 1) ** 0.5  # TODO: Might put that into function an cache.

            return 1 / 3. * fw * self.N0 * (
                - ( (chimax - chimin) * (cosLmax + cosLmax ** 2 + cosLmax ** 3 / 3 - 1 / 3. * cosLmin * (3 + cosLmin * (3 + cosLmin))) * (-3 * cosVmax + cosVmax ** 3 + 3 * cosVmin - cosVmin ** 3) * self.Hminus(w) ** 2 )
                - 1 / 3. * (-3 * cosLmax + cosLmax ** 3 + 3 * cosLmin - cosLmin ** 3) * (-3 * cosVmax + cosVmax ** 3 + 3 * cosVmin - cosVmin ** 3) * (np.sin(2 * chimax) - np.sin(2 * chimin)) * self.Hminus(w) * self.Hplus(w)
                - 1 / 3. * (chimax - chimin) * (cosLmax - cosLmin) * (3 + cosLmax ** 2 + cosLmax * (-3 + cosLmin) + (-3 + cosLmin) * cosLmin) * (-3 * cosVmax + cosVmax ** 3 + 3 * cosVmin - cosVmin ** 3) * self.Hplus(w) ** 2
                + 2 / 3. * (-(1 - cosVmax ** 2) ** 0.5 + cosVmax ** 2 * (1 - cosVmax ** 2) ** 0.5 + (1 - cosVmin ** 2) ** 0.5 - cosVmin ** 2 * (1 - cosVmin ** 2) ** 0.5) 
                    * (-2 * (1 - cosLmax ** 2) ** 0.5 
                        + 3 * cosLmax * (1 - cosLmax ** 2) ** 0.5
                        + 2 * cosLmax ** 2 * (1 - cosLmax ** 2) ** 0.5 
                        + 2 * (1 - cosLmin ** 2) ** 0.5 
                        - 3 * cosLmin * (1 - cosLmin ** 2) ** 0.5
                        - 2 * cosLmin ** 2 * (1 - cosLmin ** 2) ** 0.5
                        + 3 * np.arcsin(cosLmax)
                        - 3 * np.arcsin(cosLmin)
                    ) * (np.sin(chimax) - np.sin(chimin)) * self.Hminus(w) * self.Hzero(w)
                + 2 / 3. * (-(1 - cosVmax ** 2) ** 0.5 + cosVmax ** 2 * (1 - cosVmax ** 2) ** 0.5 + (1 - cosVmin ** 2) ** 0.5 - cosVmin ** 2 * (1 - cosVmin ** 2) ** 0.5)
                    * (-2 * (1 - cosLmax ** 2) ** 0.5
                        + cosLmax * (-3 + 2 * cosLmax) * (1 - cosLmax ** 2) ** 0.5
                        + 2 * (1 - cosLmin ** 2) ** 0.5
                        + (3 - 2 * cosLmin) * cosLmin * (1 - cosLmin ** 2) ** 0.5
                        - 3 * np.arcsin(cosLmax)
                        + 3 * np.arcsin(cosLmin)
                    ) * (np.sin(chimax) - np.sin(chimin)) * self.Hplus(w) * self.Hzero(w)
                - 4 / 3. * (chimax - chimin) * (-3 * cosLmax + cosLmax ** 3 + 3 * cosLmin - cosLmin ** 3) * (cosVmax ** 3 - cosVmin ** 3) * self.H0(w) ** 2
            )

        # --- Integrated Rates 1D ---

        def DGamma_Dw(self, wmin: float, wmax: float) -> float:
            assert self.w_min <= wmin < wmax <= self.w_max
            return scipy.integrate.quad(lambda w: dGamma_dw_DcosL_DcosV_Dchi(w, self.cosL_min, self.cosL_max, self.cosV_min, self.cosV_max, self.chi_min, self.chi_max), wmin, wmax)[0]


        def DGamma_Dw_DcosL(self, wmin: float, wmax: float, cosLmin: float, cosLmax: float) -> float:
            assert self.w_min <= wmin < wmax <= self.w_max
            assert self.cosL_min <= cosLmin < cosLmax <= self.cosL_max
            return scipy.integrate.quad(lambda w: dGamma_dw_DcosL_DcosV_Dchi(w, cosLmin, cosLmax, self.cosV_min, self.cosV_max, self.chi_min, self.chi_max), wmin, wmax)[0]


        def DGamma_Dw_DcosV(self, wmin: float, wmax: float, cosVmin: float, cosVmax: float) -> float:
            assert self.w_min <= wmin < wmax <= self.w_max
            assert self.cosV_min <= cosVmin < cosVmax <= self.cosV_max
            return scipy.integrate.quad(lambda w: dGamma_dw_DcosL_DcosV_Dchi(w, self.cosL_min, self.cosL_max, cosVmin, cosVmax, self.chi_min, self.chi_max), wmin, wmax)[0]


        def DGamma_Dw_Dchi(self, wmin: float, wmax: float, chimin: float, chimax: float) -> float:
            assert self.w_min <= wmin < wmax <= self.w_max
            assert self.chi_min <= chimin < chimax <= self.chi_max
            return scipy.integrate.quad(lambda w: dGamma_dw_DcosL_DcosV_Dchi(w, self.cosL_min, self.cosL_max, self.cosV_min, self.cosV_max, chimin, chimax), wmin, wmax)[0]


        def DGamma_DcosL_DcosV(self, cosLmin: float, cosLmax: float, cosVmin: float, cosVmax: float) -> float:
            assert self.cosL_min <= cosLmin < cosLmax <= self.cosL_max
            assert self.cosV_min <= cosVmin < cosVmax <= self.cosV_max
            return scipy.integrate.quad(lambda w: dGamma_dw_DcosL_DcosV_Dchi(w, cosLmin, cosLmax, cosVmin, cosVmax, self.chi_min, self.chi_max), self.w_min, self.w_max)[0]


        def DGamma_DcosL_Dchi(self, cosLmin: float, cosLmax: float, chimin: float, chimax: float) -> float:
            assert self.cosL_min <= cosLmin < cosLmax <= self.cosL_max
            assert self.chi_min <= chimin < chimax <= self.chi_max
            return scipy.integrate.quad(lambda w: dGamma_dw_DcosL_DcosV_Dchi(w, cosLmin, cosLmax, self.cosV_min, self.cosV_max, chimin, chimax), self.w_min, self.w_max)[0]


        def DGamma_DcosL_Dchi(self, cosVmin: float, cosVmax: float, chimin: float, chimax: float) -> float:
            assert self.cosV_min <= cosVmin < cosVmax <= self.cosV_max
            assert self.chi_min <= chimin < chimax <= self.chi_max
            return scipy.integrate.quad(lambda w: dGamma_dw_DcosL_DcosV_Dchi(w, self.cosL_min, self.cosL_max, cosVmin, cosVmax, chimin, chimax), self.w_min, self.w_max)[0]