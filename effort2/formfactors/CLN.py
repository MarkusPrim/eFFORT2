from effort2.formfactors.HelicityAmplitudes import FormFactorHQETBToP, FormFactorHQETBToV


class BToDCLN(FormFactorHQETBToP):
    """_summary_

    Reference: https://link.aps.org/accepted/10.1103/PhysRevD.94.094008

    Args:
        FormFactorHQETBToP (_type_): _description_
    """

    def __init__(
            self, 
            m_B: float, 
            m_P: float, 
            G_1: float,
            rho2: float,
            m_L: float = 0
            ) -> None:
        super().__init__(m_B, m_P, m_L)
        self.G_1 = G_1
        self.rho2 = rho2


    def fzero(self, w: float) -> float:
        r = self.r 
        return self.fplus(w) * (4 * r) / (1 + r) ** 2 * (1 + w) / 2 * 1.0036 * (1 - 0.0068 * (w - 1) + 0.0017 * (w - 1) ** 2 - 0.0013 * (w - 1) ** 3)


    def fplus(self, w: float) -> float:
        r = self.r 
        z = self.z(w)
        return (1 + r) / 2 / r ** 0.5 * self.G(z)


    def G(self, z: float) -> float:
        rho2 = self.rho2
        return self.G_1 * (1 - 8 * rho2 * z + (51 * rho2 - 10) * z ** 2 - (252 * rho2 - 84) * z ** 3)


    def __str__(self):
        return f"""CLN B --> D Expansion coefficients
G_1  = {self.G_1}
rho2  = {self.rho2}
"""


class BToDStarCLN(FormFactorHQETBToV):

    def __init__(
        self,
        m_B: float,
        m_V: float,
        h_A1_1: float,
        rho2: float,
        R1_1: float, 
        R2_1: float,
        m_L: float = 0,
        R0_1: float = 0, 
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
        super().__init__(m_B, m_V, m_L)
        self.h_A1_1 = h_A1_1
        self.rho2 = rho2
        self.R0_1 = R0_1
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


    def R0(self, w) -> None:
        """From https://arxiv.org/pdf/1203.2654.pdf

        Returns:
            _type_: _description_
        """
        return self.R0_1 - 0.11 * (w - 1) + 0.01 * (w - 1) ** 2


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


    def __str__(self):
        return f"""CLN B --> D* Expansion coefficients
h_A1_1  = {self.h_A1_1}
rho2  = {self.rho2}
R0_1  = {self.R0_1}
R1_1  = {self.R1_1}
R2_1  = {self.R2_1}"""

