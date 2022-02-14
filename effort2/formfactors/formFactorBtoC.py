import abc
import numpy as np
import numba as nb
import functools
import uncertainties.unumpy as unp
import scipy.misc

from effort2.math.functions import diLog
from effort2.formfactors.formFactorBase import FormFactor


class FormFactorBToC(FormFactor):
    r"""This class defines the interface for any form factor parametrization to be used in conjunction with the rate implementations
for $B \to P \ell \nu_\ell$ and $B \to V \ell \nu_\ell$ decays, where P stands for Pseudoscalar and V stands for Vector heavy mesons.
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


    #@functools.lru_cache()
    @staticmethod
    @nb.njit(cache=True)
    def z(w: float) -> float:
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
        The order for the expansion can be chosen arbitrarily.

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
        return self._z_p(m_pole, self.m_B, self.m_M)


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


class BLPRXP:

    def __init__(
        self,
        mB: float = 5.28,
        mD: float = 1.87,
        mDs: float = 2.01,
        mBBar: float = 5.313,
        mDBar: float = 1.973,
        ml: float = 0,
        alpha_s: float = 0.27,
        Vcb: float = 1,
        RhoSq: float = 1.2,
        Cur: float = 80,
        chi21: float = 0,
        chi2p: float = 0,
        chi3p: float = 0,
        eta1: float = 0,
        etap: float = 0,
        mb1S: float = 4.71,
        dmbc: float = 3.4,
        beta21: float = 0,
        beta3p: float = 0,
        phi1p: float = 0,
        la2: float = 0,
        ):

        self.als = alpha_s
        self.ash =  self.als/np.pi        
        self.ml = ml
        
        # Epsilon for numerical derivative
        self.eps = 10**-6
        
        self.mB = mB
        self.mD = mD
        self.mDs = mDs
        self.rD = mD / mB
        self.fDs = 2 * np.sqrt(mB * mDs) / (mB + mDs)
        
        self.mBBar = mBBar
        self.mDBar = mDBar
        
        # Fit Parameters
        self.Vcb = Vcb
        self.RhoSq = RhoSq
        self.Cur = Cur
        self.chi21 = chi21
        self.chi2p = chi2p
        self.chi3p = chi3p
        self.eta1 = eta1
        self.etap = etap
        self.mb1S = mb1S
        self.dmbc = dmbc
        self.beta21 = beta21
        self.beta3p = beta3p
        self.phi1p = phi1p
        self.la2 = la2

        # Now let's calculate some of the variables we need, note that these are just placeholders
        # for what we will have to initialize with the uncertainty class
        
        # Calculate additional parameters
        
        # Quark masses
        corr1S = 2.*(self.als/3. * 0.85)**2
        _dmbc = self.dmbc
        mb_0 = self.mb1S
        mb_1 = self.mb1S * corr1S
        mc_0 = self.mb1S - self.dmbc
        mc_1 = self.mb1S * corr1S
        LambdaBar_0 = (mb_0 * self.mBBar - mc_0 * self.mDBar)/_dmbc - (2.*mb_0-_dmbc)
        LambdaBar_1 = (mb_1 * self.mBBar - mc_1 * self.mDBar)/_dmbc - 2.*mb_1
        lambda1_0 = 2.* mb_0 * (mb_0-_dmbc)/_dmbc * (self.mBBar-self.mDBar-_dmbc)  
        
        #self.La1S = LambdaBar_0
        #self.mc1S = mc_0
                
        self.eb = LambdaBar_0/(2*mb_0)
        self.ec = LambdaBar_0/(2*mc_0)
        self.z = mc_0 / mb_0
        
        # Renormalon corrections
        self.corrb = (LambdaBar_1/LambdaBar_0 - mb_1/mb_0)*self.eb
        self.corrc = (LambdaBar_1/LambdaBar_0 - mc_1/mc_0)*self.ec
        
        # l1hat and l2hat
        self.la1h = lambda1_0/(LambdaBar_0 * LambdaBar_0)
        self.la2h = self.la2/(LambdaBar_0 * LambdaBar_0)

        self.eb2 = self.eb**2
        self.ec2 = self.ec**2
        self.eceb = self.ec * self.eb
        

    # --------------------------------------------------------------------------------------------        
    def Gamma(self,dG):
        return 0      
        
    # ------------------------------------------------------------------------------------------------        
    def dGamma(self,Hp,Hm,H0,HS,ml,p,q2):
        self.GF = 1.16637*10**(-5)
        rate = self.Vcb**2*self.GF**2/(96*np.pi**3 *self.mB**2)*(1 - ml**2/q2)**2 * p * q2 * ( (Hp**2 + Hm**2 + H0**2)*( 1 + ml**2/(2*q2) ) +  3*ml**2/(2*q2)*HS**2)
        return rate
    
    def dGammaD(self,w,ml=0):
        q2 = self.q2D(w)
        pD = self.pD(q2)
        return 2*self.mB*self.mDs * self.dGamma(0,0,self.H0D(w),self.HsD(w),ml,pD,q2)
        
    def dGammaDs(self,w,ml=0):
        q2 = self.q2Ds(w)
        pDs = self.pDs(q2)
        return 2*self.mB*self.mDs * self.dGamma(self.HpDs(w),self.HmDs(w),self.H0Ds(w),self.HsDs(w),ml,pDs,q2)
    
        
    # ------------------------------------------------------------------------------------------------        

    def pD(self,q2):
        return np.sqrt( ( (self.mB**2 + self.mD**2 - q2)/(2*self.mB) )**2 - self.mD**2 )
        
    def pDs(self,q2):
        return np.sqrt( ( (self.mB**2 + self.mDs**2 - q2)/(2*self.mB) )**2 - self.mDs**2 )
        
    def q2D(self,w):
        return -(2 * self.mB * self.mD * w - self.mB**2 - self.mD**2)
    
    def q2Ds(self,w):
        return -(2 * self.mB * self.mDs * w - self.mB**2 - self.mDs**2)
    
    def wD(self,q2):
        return (self.mB**2 + self.mD**2 - q2)/(2*self.mB*self.mD)
    
    def wDs(self,q2):
        return (self.mB**2 + self.mDs**2 - q2)/(2*self.mB*self.mDs)
    
    # ------------------------------------------------------------------------------------------------        
    # D Helicity Amplitudes
    
    def H0D(self,w):
        return np.sqrt(self.mB*self.mD)*(self.mB+self.mD)/np.sqrt(self.q2D(w))*np.sqrt(w**2-1.)*self.V1D(w)

    def HsD(self,w):
        return np.sqrt(self.mB*self.mD)*(self.mB-self.mD)/np.sqrt(self.q2D(w))*(w+1)*self.S1D(w)
    
    # ------------------------------------------------------------------------------------------------        
    # D* Helicity Amplitudes
    
    def HpDs(self,w):
        return (self.mB+self.mDs)*self.A1Ds(w) - 2*self.mB/(self.mB+self.mDs)*self.pDs(self.q2Ds(w))*self.VDs(w)
        
    def HmDs(self,w):   
        return (self.mB+self.mDs)*self.A1Ds(w) + 2*self.mB/(self.mB+self.mDs)*self.pDs(self.q2Ds(w))*self.VDs(w)
    
    def H0Ds(self,w):
        return 1./(2*self.mDs*np.sqrt(self.q2Ds(w))) * ( (self.mB**2 - self.mDs**2 - self.q2Ds(w))*(self.mB + self.mDs)*self.A1Ds(w) - (4*self.mB**2 * self.pDs(self.q2Ds(w))**2)/(self.mB + self.mDs)*self.A2Ds(w) )

    def HsDs(self,w):
        return (2*self.mB*self.pDs(self.q2Ds(w)))/np.sqrt(self.q2Ds(w)) * self.A0Ds(w)
    
    # ------------------------------------------------------------------------------------------------        
    # D form factors
    
    def V1D(self,w):
        return self.hp(w) - (self.mB - self.mD)/(self.mB + self.mD) * self.hm(w)

    def S1D(self,w):
        return self.hp(w) - (self.mB + self.mD)/(self.mB - self.mD) * (w-1.)/(w+1.) * self.hm(w)
       
    # ------------------------------------------------------------------------------------------------        
    # D* form factors
        
    def A1Ds(self,w):
        return (w+1.)/2. * self.fDs * self.hA1(w)

    def A2Ds(self,w):
        return self.R2(w)/self.fDs * self.hA1(w)

    def A0Ds(self,w):
        return self.R0(w)/self.fDs * self.hA1(w)
    
    def VDs(self,w):
        return self.R1(w)/self.fDs * self.hA1(w)   

    def R0(self,w):
        return self.A0(w) * self.fDs / self.hA1(w)    

    def R1(self,w):
        return self.hV(w) / self.hA1(w)
    
    def R2(self,w):
        return ( self.hA3(w) + self.mDs/self.mB * self.hA2(w) ) / self.hA1(w)
        
    def R3(self,w):
        return (self.hA3(w) - self.mDs/self.mB * self.hA2(w)) / self.hA1(w)

    def A0(self,w):
        return self.A3(w) + self.q2Ds(w)/(4*self.mB*self.mDs)*np.sqrt(self.mB/self.mDs) * ( self.hA3(w) - self.mDs/self.mB * self.hA2(w) )
    
    def A3(self,w):
        return (self.mB + self.mDs)/(2*np.sqrt(self.mB*self.mDs)) *( self.mB/(self.mB + self.mDs) *(w+1) * self.hA1(w) - (self.mB - self.mDs)/(2*self.mDs) * ( self.hA3(w) + self.mDs/self.mB * self.hA2(w) ) )
    
    # ------------------------------------------------------------------------------------------------        

    def PrintAllPars(self):
        
        print("Vcb: ", self.Vcb)
        print("RhoSq: ", self.RhoSq)
        print("Cur: ", self.Cur)
        print("chi21: ", self.chi21)
        print("chi2p: ", self.chi2p)
        print("chi3p: ", self.chi3p)
        print("eta1: ", self.eta1)
        print("etap: ", self.etap)
        print("mb1S: ", self.mb1S)
        print("dmbc: ", self.dmbc)
        print("beta21: ", self.beta21)
        print("beta3p: ", self.beta3p)
        print("phi1p: ", self.phi1p)
        print("la2: ", self.la2)
        print("z:", self.z)
        print("eb:", self.eb)
        print("ec:", self.ec)
    
    # Kinematics
    def wmin(self):
        return 1.0
    
    def wmaxD(self,ml=0):
        return (self.mB**2 + self.mD**2 - ml**2)/(2*self.mB*self.mD)
    
    def wmaxDs(self,ml=0):
        return (self.mB**2 + self.mDs**2 - ml**2)/(2*self.mB*self.mDs)    
        
    # Leading IW Function 
    def xi(self,w):
    
        # optimized expansion
        a = ( (1 + self.rD)/(2*np.sqrt(self.rD)) )**0.5
        zs = ( (w+1)**0.5 - (2)**0.5*a )/( (w+1)**0.5 + (2)**0.5*a )
        zsn = ( 1 - a )/( 1 + a )

        xi1 = 1 - 8*a**2*self.RhoSq*zsn + self.Cur*zsn**2
                
        return (1 - 8*a**2*self.RhoSq*zs + self.Cur*zs**2 ) / xi1
      
    # ------------------------------------------------------------------------------------------------        

    def hph(self,w):
        return self.hph_1(w) + self.hph_2(w) + self.hph_as1(w)
                
    def hmh(self,w):
        return self.hmh_1(w) + self.hmh_2(w) + self.hmh_as1(w)
        
    def hVh(self,w):
        return self.hVh_1(w) + self.hVh_2(w) + self.hVh_as1(w)
        
    def hA1h(self,w):
        return self.hA1h_1(w) + self.hA1h_2(w) + self.hA1h_as1(w)
        
    def hA2h(self,w):
        return self.hA2h_1(w) + self.hA2h_2(w) + self.hA2h_as1(w)

    def hA3h(self,w):
        return self.hA3h_1(w) + self.hA3h_2(w) + self.hA3h_as1(w)
        
    
    # ------------------------------------------------------------------------------------------------        

    def hph_1(self,w):
        return 1 + self.ash*(self.CV1(w) + (w+1)/2*(self.CV2(w)+self.CV3(w))) + self.eb*self.L1_1(w) + self.ec*self.L1_1(w)
                
    def hmh_1(self,w):
        return self.ash*(w+1)/2*(self.CV2(w) - self.CV3(w)) + self.ec*self.L4_1(w) - self.eb*self.L4_1(w) - (self.corrc - self.corrb)  
        
    def hVh_1(self,w):
        return 1 + self.ash*self.CV1(w) + self.ec*(self.L2_1(w) - self.L5_1(w)) + self.eb*(self.L1_1(w) - self.L4_1(w)) + self.corrc + self.corrb
        
    def hA1h_1(self,w):
        return 1 + self.ash*self.CA1(w) + self.ec*(self.L2_1(w) - self.L5_1(w)*(w-1)/(w+1)) + self.eb*(self.L1_1(w) - self.L4_1(w)*(w-1)/(w+1)) + (self.corrc+self.corrb) * (w-1.)/(w+1.)
        
    def hA2h_1(self,w):
        return self.ash*self.CA2(w) + self.ec*(self.L3_1(w) + self.L6_1(w)) - 2./(w+1.) * self.corrc

    def hA3h_1(self,w):
        return 1 + self.ash*(self.CA1(w) + self.CA3(w)) + self.ec*(self.L2_1(w) - self.L3_1(w) + self.L6_1(w) - self.L5_1(w)) + self.eb*(self.L1_1(w) - self.L4_1(w)) + self.corrc * (w-1.)/(w+1.) + self.corrb
          
    # ------------------------------------------------------------------------------------------------        

    def hph_2(self,w):
        return self.ec2 * self.L1_2(w) + self.eb2 * self.L1_2(w) - self.eceb * self.M8(w)
    
    def hmh_2(self,w):
        return self.ec2 * self.L4_2(w) + self.eb2 * self.L4_2(w)
     
    def hVh_2(self,w):
        return self.ec2 * (self.L2_2(w) - self.L5_2(w)) + self.eb2 * (self.L2_2(w) - self.L5_2(w)) + self.eceb * self.M9(w)
    
    def hA1h_2(self,w):
        wm1Owp1 = (w - 1.)/(w + 1.)
        return self.ec2 * (self.L2_2(w) - self.L5_2(w) * wm1Owp1) + self.eb2 * (self.L2_2(w) - self.L5_2(w) * wm1Owp1) + self.eceb * self.M9(w)
        
    def hA2h_2(self,w):
        return self.ec2 * (self.L6_2(w) + self.L3_2(w)) + self.eb2 * (self.L6_2(w) + self.L3_2(w)) - self.eceb * self.M10(w)
    
    def hA3h_2(self,w):
        return self.ec2 * (self.L2_2(w) - self.L3_2(w) + self.L6_2(w) - self.L5_2(w)) + self.eb2 * (self.L2_2(w) - self.L3_2(w) + self.L6_2(w) - self.L5_2(w)) + self.eceb * (self.M9(w) + self.M10(w))

    
    # ------------------------------------------------------------------------------------------------
    # HQE Form factors
    
    def hp(self,w):
        return self.hph(w) * self.xi(w)
    
    def hm(self,w):
        return self.hmh(w) * self.xi(w)
    
    def hV(self,w):
        return self.hVh(w) * self.xi(w)
    
    def hA1(self,w):
        return self.hA1h(w) * self.xi(w)
    
    def hA2(self,w):
        return self.hA2h(w) * self.xi(w)
    
    def hA3(self,w):
        return self.hA3h(w) * self.xi(w)    
    
    
    # ------------------------------------------------------------------------------------------------
    # as functions
    
    def CV1(self,w):    
        
        z = self.z    
        wSq = w**2
        zSq = z**2
        sqrt1wSq = np.sqrt(wSq - 1)
        lnz = unp.log(z)
        polylog1 = diLog((2 + 2*w*(-w + sqrt1wSq)))
        polylog2 = diLog((2 - 2*w*(w + sqrt1wSq)))
        polylog3 = diLog((1 - w*z + sqrt1wSq*z))
        polylog4 = diLog((1 - (w + sqrt1wSq)*z))
        
        return (-2*np.log(w + sqrt1wSq)*(-((1 + w)*(-1 + (-1 + 3*w - z)*z)) + w*(-1 + 2*w*z - zSq)*lnz) + sqrt1wSq*(4 + 4*z*(-2*w + z) - (-1 + zSq)*lnz) +w*(-1 + 2*w*z - zSq)*(-polylog1 + polylog2 + 2*polylog3 - 2*polylog4))/(3.*sqrt1wSq*(-1 + 2*w*z - zSq))       

    def CV2(self,w):
          
        z = self.z        
        wSq = w**2
        zSq = z**2
        sqrt1wSq = np.sqrt(wSq - 1)
        zXX = (1 - 2*w*z + zSq)**2
        lnz = unp.log(z)
        
        return (-2*((2 - z*(-1 + wSq*(2 - 4*z) + zSq + w*(5 + (-2 + z)*z)))*np.log(w + sqrt1wSq) + sqrt1wSq*((-1 + z)*(1 - 2*w*z + zSq) +z*(3 + z*(2 + z) - 2*w*(1 + 2*z))*lnz)))/(3.*sqrt1wSq*zXX)       
           
    def CV3(self,w): 
                 
        z = self.z
        wSq = w**2
        zSq = z**2
        sqrt1wSq = np.sqrt(wSq - 1)
        zXX = (1 - 2*w*z + zSq)**2
        lnz = unp.log(z)   
        
        return (2*z*((1 + w - 2*w*(1 + 2*w)*z + (-1 + w*(5 + 2*w))*zSq - 2*pow(z,3.))*np.log(w + sqrt1wSq) + sqrt1wSq*((-1 + z)*(1 - 2*w*z + zSq) +(1 + z*(2 + 3*z - 2*w*(2 + z)))*lnz)))/(3.*sqrt1wSq*zXX)
        
        
    def CA1(self,w):
        
        z = self.z
        wSq = w**2
        zSq = z**2
        sqrt1wSq = np.sqrt(wSq - 1)
        lnz = unp.log(z)
        polylog1 = diLog((2 + 2*w*(-w + sqrt1wSq)))
        polylog2 = diLog((2 - 2*w*(w + sqrt1wSq)))
        polylog3 = diLog((1 - w*z + sqrt1wSq*z))
        polylog4 = diLog((1 - (w + sqrt1wSq)*z))
        
        return (-2*np.log(w + sqrt1wSq)*(-((-1 + w)*(-1 + z + 3*w*z - zSq)) + w*(-1 + 2*w*z - zSq)*lnz) + sqrt1wSq*(4 + 4*z*(-2*w + z) - (-1 + zSq)*lnz) - w*(-1 + 2*w*z - zSq)*(polylog1 - polylog2 - 2*polylog3 + 2*polylog4))/(3.*sqrt1wSq*(-1 + 2*w*z - zSq))       
        
        
    def CA2(self,w):
        
        z = self.z
        wSq = w**2
        zSq = z**2
        sqrt1wSq = np.sqrt(wSq - 1)
        zXX = (1 - 2*w*z + zSq)**2
        lnz = unp.log(z)   

        return (-2*((2 + z*(-1 + zSq + wSq*(2 + 4*z) - w*(5 + z*(2 + z))))* np.log(w + sqrt1wSq) + sqrt1wSq*((1 + z)*(1 - 2*w*z + zSq) +z*(3 + w*(2 - 4*z) + (-2 + z)*z)*lnz)))/(3.*sqrt1wSq*zXX)        
        
    def CA3(self,w):  
                
        z = self.z
        wSq = w**2
        zSq = z**2
        sqrt1wSq = np.sqrt(wSq - 1)
        zXX = (1 - 2*w*z + zSq)**2
        lnz = unp.log(z)

        return (2*z*((1 + 2*wSq*z*(2 + z) + zSq*(-1 + 2*z) - w*(1 + z*(2 + 5*z)))*np.log(w + sqrt1wSq) + sqrt1wSq*((1 + z)*(1 - 2*w*z + zSq) +(-1 + z*(2 + 4*w - 3*z - 2*w*z))*lnz)))/(3.*sqrt1wSq*zXX)
    
    # ------------------------------------------------------------------------------------------------
    # Derivatives of as functions

    def derCA1(self,w):
        return scipy.misc.derivative(self.CA1, w, self.eps)

    def derCA2(self,w):
        return scipy.misc.derivative(self.CA2, w, self.eps)

    def derCA3(self,w):
        return scipy.misc.derivative(self.CA3, w, self.eps)
    
    def derCV1(self,w):
        return scipy.misc.derivative(self.CV1, w, self.eps)

    def derCV2(self,w):
        return scipy.misc.derivative(self.CV2, w, self.eps)

    def derCV3(self,w):
        return scipy.misc.derivative(self.CV3, w, self.eps)
    

    # ------------------------------------------------------------------------------------------------    
    # L master functions
    
    def L1b(self,w):
        return self.L1_1(w) + self.eb*self.L1_2(w)

    def L1c(self,w):
        return self.L1_1(w) + self.ec*self.L1_2(w)
    
    def L2b(self,w):
        return self.L2_1(w) + self.eb*self.L2_2(w)

    def L2c(self,w):
        return self.L2_1(w) + self.ec*self.L2_2(w)    
    
    def L3b(self,w):
        return self.L3_1(w) + self.eb*self.L3_2(w)

    def L3c(self,w):
        return self.L3_1(w) + self.ec*self.L3_2(w)   
    
    def L4b(self,w):
        return self.L4_1(w) + self.eb*self.L4_2(w)

    def L4c(self,w):
        return self.L4_1(w) + self.ec*self.L4_2(w)   
    
    def L5b(self,w):
        return self.L5_1(w) + self.eb*self.L5_2(w)

    def L5c(self,w):
        return self.L5_1(w) + self.ec*self.L5_2(w)    
    
    def L6b(self,w):
        return self.L6_1(w) + self.eb*self.L6_2(w)

    def L6c(self,w):
        return self.L6_1(w) + self.ec*self.L6_2(w)        
    
        

    # ------------------------------------------------------------------------------------------------    
    # L functions for 1/m corrections
    
    def L1_1(self,w):
        return -4.*(w-1)*(self.chi21 + (w-1.)*self.chi2p)+12.*self.chi3p*(w-1.)
        
    def L2_1(self,w):
        return -4.*self.chi3p*(w-1.)
        
    def L3_1(self,w):
        return 4.*(self.chi21 + (w-1.)*self.chi2p)

    def L4_1(self,w):
        return 2.*(self.eta1 + self.etap*(w-1.)) - 1
        
    def L5_1(self,w):
        return -1.0
    
    def L6_1(self,w):
        return -2.*(1+(self.eta1 + self.etap*(w-1.)))/(w+1.)
       
    # ------------------------------------------------------------------------------------------------
    # L functions for 1/m**2 corrections
    
    def beta1(self,w):
        return self.la1h/4.

    def beta2(self,w):
        return self.beta21

    def beta3(self,w):
        return (self.la2h)/8. + self.beta3p*(w-1.)
    
    def phi1(self,w):
        return (self.la1h/3. - self.la2h/2.)/2.  +  self.phi1p*(w-1.)
    
    def L1_2(self,w):        
        return 2.* self.beta1(w) + 4.*(3.* self.beta3(w) - (w-1.)* self.beta2(w))
         
    def L2_2(self,w):
        return 2* self.beta1(w) - 4 * self.beta3(w)
        
    def L3_2(self,w):
        return 4*self.beta2(w)
        
    def L4_2(self,w):
        return 3.*self.la2h + 2. * (w+1.) * self.phi1(w)
        
    def L5_2(self,w):
        return self.la2h + 2 * (w+1.) * self.phi1(w)
        
    def L6_2(self,w):
        return 4 * self.phi1(w)
    
    # ------------------------------------------------------------------------------------------------
    # M functions for 1/mb * 1/mc  corrections
    
    def eta(self,w):
        return self.eta1 + self.etap*(w-1)

    #def phi1p(self,w):
        #return self.phi1p
    
    def M8(self,w):
        return (self.la1h+ 6.*self.la2h/(w + 1.) -2. * (w - 1.) * self.phi1(w) - 2. * (2.* self.eta(w) - 1.) * (w - 1.)/(w + 1.))
    
    def M9(self,w):
        return 3.*self.la2h/(w+1.) + 2. * self.phi1(w) - (2.* self.eta(w) - 1.) * (w - 1.)/(w + 1.)
    
    def M10(self,w):
        return self.la1h/3. - self.la2h * (w+4.)/(2.*(w+1.)) + 2. * (w + 2.) * self.phi1p - (2.* self.eta(w) - 1.)/(w + 1.)
    
    # ------------------------------------------------------------------------------------------------    
    # as x 1/m corrections to h_{+,-,V,A1-A3} functions 
    
    def cMagB(self):
        return 3./2.*(0.5*unp.log(self.z) + 13./9.)        
        
    def cMagC(self):
        return -3./2.*(0.5*unp.log(self.z) - 13./9.)      
            
    def hph_as1(self,w):
        
        aseb = self.ash * self.eb
        asec = self.ash * self.ec
        
        cmagb = self.cMagB()
        cmagc = self.cMagC()
        
        value = aseb * (cmagb*self.L1_1(w) + self.L1_1(w)*self.CV1(w) - self.L5_1(w)*(-1 + w)*self.CV2(w) + ((1 + w)*(self.L1_1(w) - (self.L4_1(w)*(-1 + w))/(1 + w))*(self.CV2(w) + self.CV3(w)))/2. + 2*(-1 + w)*(self.derCV1(w) + ((1 + w)*(self.derCV2(w) + self.derCV3(w)))/2.))
        value += asec * (cmagc*self.L1_1(w) + self.L1_1(w)*self.CV1(w) - self.L5_1(w)*(-1 + w)*self.CV3(w) + ((1 + w)*(self.L1_1(w) - (self.L4_1(w)*(-1 + w))/(1 + w))*(self.CV2(w) + self.CV3(w)))/2. + 2*(-1 + w)*(self.derCV1(w) + ((1 + w)*(self.derCV2(w) + self.derCV3(w)))/2.))

        return value
    
    def hmh_as1(self,w):
        
        aseb = self.ash * self.eb
        asec = self.ash * self.ec
        
        value = aseb * (-(self.L4_1(w)*self.CV1(w)) - self.L5_1(w)*(1 + w)*self.CV2(w) + ((1 + w)*((self.L1_1(w) - (self.L4_1(w)*(-1 + w))/(1 + w))*(self.CV2(w) - self.CV3(w)) + 2*(-1 + w)*(self.derCV2(w) - self.derCV3(w))))/2.)    
        value += asec * (self.L4_1(w)*self.CV1(w) + self.L5_1(w)*(1 + w)*self.CV3(w) + ((1 + w)*((self.L1_1(w) - (self.L4_1(w)*(-1 + w))/(1 + w))*(self.CV2(w) - self.CV3(w)) + 2*(-1 + w)*(self.derCV2(w) - self.derCV3(w))))/2.)
        
        return value   
    
    def hVh_as1(self,w):
        
        aseb = self.ash * self.eb
        asec = self.ash * self.ec
        
        cmagb = self.cMagB()
        cmagc = self.cMagC()
                
        value = aseb * (cmagb*self.L1_1(w) + (self.L1_1(w) - self.L4_1(w))*self.CV1(w) - (self.L4_1(w) - self.L5_1(w))*self.CV2(w) + 2*(-1 + w)*self.derCV1(w))
        value += asec * (cmagc*self.L2_1(w) + (self.L2_1(w) - self.L5_1(w))*self.CV1(w) - (self.L4_1(w) - self.L5_1(w))*self.CV3(w) + 2*(-1 + w)*self.derCV1(w))
        
        return value
    
    def hA1h_as1(self,w):
        
        aseb = self.ash * self.eb
        asec = self.ash * self.ec
        
        cmagb = self.cMagB()
        cmagc = self.cMagC()
        
        value = aseb * (cmagb*self.L1_1(w) + (self.L1_1(w) - (self.L4_1(w)*(-1 + w))/(1 + w))*self.CA1(w) + ((self.L4_1(w) - self.L5_1(w))*(-1 + w)*self.CA2(w))/(1 + w) + 2*(-1 + w)*self.derCA1(w))
        value += asec * (cmagc*self.L2_1(w) + (self.L2_1(w) - (self.L5_1(w)*(-1 + w))/(1 + w))*self.CA1(w) + ((self.L4_1(w) - self.L5_1(w))*(-1 + w)*self.CA3(w))/(1 + w) + 2*(-1 + w)*self.derCA1(w))
    
        return value  
    
    def hA2h_as1(self,w):
        
        aseb = self.ash * self.eb
        asec = self.ash * self.ec
        
        cmagc = self.cMagC()
        
        value = aseb * ((self.L1_1(w) - self.L4_1(w) - 2*self.L5_1(w))*self.CA2(w) - ((self.L4_1(w) - 3*self.L5_1(w))*self.CA2(w))/(1 + w) + 2*(-1 + w)*self.derCA2(w))
        value += asec * (cmagc*self.L3_1(w) + (self.L3_1(w) + self.L6_1(w))*self.CA1(w) + (self.L2_1(w) + self.L5_1(w) + self.L3_1(w)*(-1 + w) - self.L6_1(w)*(1 + w))*self.CA2(w) - ((self.L4_1(w) - 3*self.L5_1(w))*self.CA3(w))/(1 + w) + 2*(-1 + w)*self.derCA2(w))    
        
        return value  
    
    def hA3h_as1(self,w):
        
        aseb = self.ash * self.eb
        asec = self.ash * self.ec
        
        cmagb = self.cMagB()
        cmagc = self.cMagC()
        
        value = aseb * (cmagb*self.L1_1(w) + 2*self.L5_1(w)*self.CA2(w) + ((self.L4_1(w) - 3*self.L5_1(w))*w*self.CA2(w))/(1 + w) + (self.L1_1(w) - self.L4_1(w))*(self.CA1(w) + self.CA3(w)) + 2*(-1 + w)*(self.derCA1(w) + self.derCA3(w)))
        value += asec * (cmagc*(-self.L3_1(w) + self.L2_1(w)) + (self.L2_1(w) - self.L3_1(w) - self.L5_1(w) + self.L6_1(w))*self.CA1(w) + ((self.L4_1(w) - 3*self.L5_1(w))*w*self.CA3(w))/(1 + w) + (self.L2_1(w) + self.L5_1(w) + self.L3_1(w)*(-1 + w) - self.L6_1(w)*(1 + w))*self.CA3(w) + 2*(-1 + w)*(self.derCA1(w) + self.derCA3(w)))
        
        return value     