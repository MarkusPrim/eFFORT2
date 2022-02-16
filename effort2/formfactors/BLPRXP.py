import numpy as np
from uncertainties import unumpy as unp
import scipy.misc

from effort2.math.functions import diLog
from effort2.formfactors.kinematics import Kinematics

class BLPRXP:

    def __init__(
        self,
        m_B: float = 5.28,
        m_D: float = 1.87,
        m_Ds: float = 2.01,
        m_L: float = 0,
        mBBar: float = 5.313,
        mDBar: float = 1.973,
        alpha_s: float = 0.27,
        Vcb: float = 1,
        RhoSq: float = 1.2,
        Cur: float = 80,
        D: float = 0,
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
        
        # Epsilon for numerical derivative
        self.eps = 10**-6
        
        self.m_B = m_B
        self.m_D = m_D
        self.m_Ds = m_Ds
        self.m_L = m_L
        self.rD = m_D / m_B
        self.fDs = 2 * np.sqrt(m_B * m_Ds) / (m_B + m_Ds)  # Equivalent to rprime
        
        self.mBBar = mBBar
        self.mDBar = mDBar
        
        # Definitions for automatic settings when doing composition
        self.m_P = m_D
        self.m_V = m_Ds

        # Fit Parameters
        self.Vcb = Vcb
        self.RhoSq = RhoSq
        self.Cur = Cur
        self.D = D
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
        corr1S = 2.*(self.als/3. * 0.8)**2
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

    # Leading IW Function
    def xi(self,w):
   
        # optimized expansion
        a = ( (1 + self.rD)/(2*np.sqrt(self.rD)) )**0.5
        zs = ( (w+1)**0.5 - (2)**0.5*a )/( (w+1)**0.5 + (2)**0.5*a )
        zsn = ( 1 - a )/( 1 + a )

        xi1 = 1 - 8*a**2*self.RhoSq*zsn + self.Cur*zsn**2 + self.D*zsn**3
               
        return (1 - 8*a**2*self.RhoSq*zs + self.Cur*zs**2 + self.D*zs**3 ) / xi1

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
        return self.ec2 * self.L4_2(w) - self.eb2 * self.L4_2(w)
     
    def hVh_2(self,w):
        return self.ec2 * (self.L2_2(w) - self.L5_2(w)) + self.eb2 * (self.L1_2(w) - self.L4_2(w)) + self.eceb * self.M9(w)
    
    def hA1h_2(self,w):
        wm1Owp1 = (w - 1.)/(w + 1.)
        return self.ec2 * (self.L2_2(w) - self.L5_2(w) * wm1Owp1) + self.eb2 * (self.L1_2(w) - self.L4_2(w) * wm1Owp1) + self.eceb * self.M9(w)
        
    def hA2h_2(self,w):
        return self.ec2 * (self.L6_2(w) + self.L3_2(w)) - self.eceb * self.M10(w)
    
    def hA3h_2(self,w):
        return self.ec2 * (self.L2_2(w) - self.L3_2(w) + self.L6_2(w) - self.L5_2(w)) + self.eb2 * (self.L1_2(w) - self.L4_2(w)) + self.eceb * (self.M9(w) + self.M10(w))
    
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


class BToDBLPRXP(BLPRXP):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kinematics = Kinematics(self.m_B, self.m_D, self.m_L)
    
    def Hzero(self,w):
        return np.sqrt(self.m_B*self.m_D)*(self.m_B+self.m_D)/np.sqrt(self.kinematics.q2(w))*np.sqrt(w**2-1.)*self.V1(w)

    def Hscalar(self,w):
        return np.sqrt(self.m_B*self.m_D)*(self.m_B-self.m_D)/np.sqrt(self.kinematics.q2(w))*(w+1)*self.S1(w)
    
    def V1(self,w):
        return self.hp(w) - (self.m_B - self.m_D)/(self.m_B + self.m_D) * self.hm(w)

    def S1(self,w):
        return self.hp(w) - (self.m_B + self.m_D)/(self.m_B - self.m_D) * (w-1.)/(w+1.) * self.hm(w)


class BToDStarBLPRXP(BLPRXP):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kinematics = Kinematics(self.m_B, self.m_Ds, self.m_L)

    def Hplus(self,w):
        return (self.m_B+self.m_Ds)*self.A1(w) - 2*self.m_B/(self.m_B+self.m_Ds)*self.kinematics.p(self.kinematics.q2(w))*self.V(w)
        
    def Hminus(self,w):   
        return (self.m_B+self.m_Ds)*self.A1(w) + 2*self.m_B/(self.m_B+self.m_Ds)*self.kinematics.p(self.kinematics.q2(w))*self.V(w)
    
    def Hzero(self,w):
        return 1./(2*self.m_Ds*np.sqrt(self.kinematics.q2(w))) * ( (self.m_B**2 - self.m_Ds**2 - self.kinematics.q2(w))*(self.m_B + self.m_Ds)*self.A1(w) - (4*self.m_B**2 * self.kinematics.p(self.kinematics.q2(w))**2)/(self.m_B + self.m_Ds)*self.A2(w) )

    def Hscalar(self,w):
        return (2*self.m_B*self.kinematics.p(self.kinematics.q2(w)))/np.sqrt(self.kinematics.q2(w)) * self.A0(w)

    
    # def A0(self,w):  # Can be inherited
    #     return self.R0(w)/self.fDs * self.hA1(w)

    def A0(self,w):
        return self.A3(w) + self.kinematics.q2(w)/(4*self.m_B*self.m_Ds)*np.sqrt(self.m_B/self.m_Ds) * ( self.hA3(w) - self.m_Ds/self.m_B * self.hA2(w) )

    def A1(self,w):  # Can be inherited
        return (w+1.)/2. * self.fDs * self.hA1(w)

    def A2(self,w):  # Can be inherited
        return self.R2(w)/self.fDs * self.hA1(w)

    def A3(self,w):
        return (self.m_B + self.m_Ds)/(2*np.sqrt(self.m_B*self.m_Ds)) *( self.m_B/(self.m_B + self.m_Ds) *(w+1) * self.hA1(w) - (self.m_B - self.m_Ds)/(2*self.m_Ds) * ( self.hA3(w) + self.m_Ds/self.m_B * self.hA2(w) ) )

    def V(self,w):  # Can be inherited
        return self.R1(w)/self.fDs * self.hA1(w)   

    def R0(self,w):
        return self.A0(w) * self.fDs / self.hA1(w)    

    def R1(self,w):
        return self.hV(w) / self.hA1(w)
    
    def R2(self,w):
        return ( self.hA3(w) + self.m_Ds/self.m_B * self.hA2(w) ) / self.hA1(w)
        
    def R3(self,w):
        return (self.hA3(w) - self.m_Ds/self.m_B * self.hA2(w)) / self.hA1(w)


if __name__ == "__main__":
    pass