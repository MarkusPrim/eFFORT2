import numpy as np
from effort2.math.functions import diLog


class DStStAlphaSCorrections:
    """
    Class to compute the C functions used in the BLR model for B -> D** form factors.
    Taken from Phys. Rev. D 95, 115008 Appendix A.
    """

    def __init__(
        self,
        m_c: float,
        m_b: float,
    ) -> None:
        self.z = m_c / m_b
        self.wz = 0.5 * (self.z + 1 / self.z)


    def r(
        self,
        w: float,
    ):
        wp = w + np.sqrt(w**2 - 1)
        return np.log(wp) / np.sqrt(w**2 - 1)


    def Omega(
        self,
        w: float,
    ):
        z = self.z
        r = self.r(w)
        wp = w + np.sqrt(w**2 - 1)
        wm = w - np.sqrt(w**2 - 1)
        return w / (2 * np.sqrt(w**2 - 1)) * (2 * diLog(1 - wm * z) - 2 * diLog(1 - wp * z) + diLog(1 - wp**2) - diLog(1 - wm**2)) - w * r * np.log(z) + 1


    def CS(
        self,
        w: float,
    ):
        z = self.z
        wz = self.wz
        r = self.r(w)
        Omega = self.Omega(w)
        return 1 / (3 * z * (w - wz)) * (2 * z * (w - wz) * Omega - (w - 1) * (z + 1) * (z + 1) * r + (z**2 -1) * np.log(z))


    def CP(
        self,
        w: float,
    ):
        z = self.z
        wz = self.wz
        r = self.r(w)
        Omega = self.Omega(w)
        return 1 / (3 * z * (w - wz)) * (2 * z * (w - wz) * Omega - (w + 1) * (z - 1) * (z - 1) * r + (z**2 -1) * np.log(z))
    

    def CV1(
        self,
        w: float,
    ):
        z = self.z
        wz = self.wz
        r = self.r(w)
        Omega = self.Omega(w)
        return 1 / (6 * z * (w - wz)) * (2 * (w + 1) * ((3 * w - 1) * z - z**2 - 1) * r + (12 * z * (wz - w) - (z**2 -1) * np.log(z)) + 4 * z * (w - wz) * Omega)
    
    def CV2(
        self,
        w: float,
    ):
        z = self.z
        wz = self.wz
        r = self.r(w)
        return -1 / (6 * z**2 * (w - wz) * (w - wz)) * (((4 * w**2 + 2 * w) * z**2 - (2 * w**2 + 5 * w -1) * z - (w + 1) * z**3 + 2) * r + z * ( 2 * (z - 1) * (wz - w) + (z**2 - (4 * w - 2) * z + (3 - 2 * w)) * np.log(z)))


    def CV3(
        self,
        w: float,
    ):
        z = self.z
        wz = self.wz
        r = self.r(w)
        return 1 / (6 * z * (w - wz) * (w - wz)) * (((2 * w**2 + 5 * w - 1) * z**2 - (4 * w**2 + 2 * w) * z - 2 * z**3 + w + 1) * r + (2 * z * (z - 1) * (wz - w) + ((3 - 2 * w) * z**2 + (2 - 4 * w) * z + 1) * np.log(z)))
    

    def CA1(
        self,
        w: float,
    ):
        z = self.z
        wz = self.wz
        r = self.r(w)
        Omega = self.Omega(w)
        return 1 / (6 * z * (w - wz)) * (2 * (w - 1) * ((3 * w + 1) * z - z**2 - 1) * r + (12 * z * (wz - w) - (z**2 - 1) * np.log(z)) + 4 * z * (w - wz) * Omega)


    def CA2(
        self,
        w: float,
    ):
        z = self.z
        wz = self.wz
        r = self.r(w)
        return -1 / (6 * z**2 * (w - wz) * (w - wz)) * (((4 * w**2 - 2 * w) * z**2 - (2 * w**2 - 5 * w -1) * z - (w - 1) * z**3 + 2) * r + z * ( 2 * (z + 1) * (wz - w) + (z**2 - (4 * w + 2) * z + (3 + 2 * w)) * np.log(z)))
    

    def CA3(
        self,
        w: float,
    ):
        z = self.z
        wz = self.wz
        r = self.r(w)
        return 1 / (6 * z * (w - wz) * (w - wz)) * ((2 * z**3 + (2 * w**2 - 5 * w -1) * z**2 + (4 * w**2 - 2 * w) * z - w + 1) * r + (2 * z * (z + 1) * (wz - w) - ((2 * w + 3) * z**2 - (4 * w + 2) * z + 1) * np.log(z)))


    def CT1(
        self,
        w: float,
    ):
        z = self.z
        wz = self.wz
        r = self.r(w)
        Omega = self.Omega(w)
        return 1 / (3 * z * (w - wz)) * ((w - 1) * ((4 * w + 2) * z - z**2 - 1) * r + (6 * z * (wz - w) - (z**2 - 1) * np.log(z)) + 2 * z * (w - wz) * Omega)
    

    def CT2(
        self,
        w: float,
    ):
        z = self.z
        wz = self.wz
        r = self.r(w)
        return 2 / (3 * z * (w - wz)) * ((1 - w * z) * r + z * np.log(z))
    

    def CT3(
        self,
        w: float,
    ):
        z = self.z
        wz = self.wz
        r = self.r(w)
        return 2 / (3 * (w - wz)) * ((w - z) * r + np.log(z))


class IWFunctions:
    """
    Class to define the Isgure-Wise functions used in the BLR model for B -> D** form factors
    """

    def __init__(self) -> None:
        pass