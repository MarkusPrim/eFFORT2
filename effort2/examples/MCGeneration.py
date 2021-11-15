from effort2.rates.BtoV import BtoV
from effort2.formfactors.formFactorBtoC import BToDStarBGL

import numpy as np
import pandas as pd


def main():
    # Values by Florian
    a = np.array([3.79139e-04, 2.69537e-02])
    b = np.array([5.49846e-04, -2.04028e-03])
    c = np.array([-4.32818e-04, 5.35029e-03])

    # Initialize form factors
    bToDStarBGL = BToDStarBGL(
        m_B=5.27963,
        m_M=2.01026,
        exp_coeff_a = a,
        exp_coeff_b = b,
        exp_coeff_c = c,
    )

    rate = BtoV(bToDStarBGL, Vcb=1, eta_EW=1, m_B=bToDStarBGL.m_B, m_V=bToDStarBGL.m_M) 
    wbins = np.linspace(rate.w_min, rate.w_max, num=11)
    cosLbins = np.linspace(rate.cosL_min, rate.cosL_max, num=11)
    cosVbins = np.linspace(rate.cosV_min, rate.cosV_max, num=11)
    chibins = np.linspace(rate.chi_min, rate.chi_max, num=11)

    tauBplus = 1.638e-12 * 1. / 6.582119e-16 / 1e-9
    eeToUpsilon4S_cross_section = 1.1e-9  # 1.1 nb
    luminosity = 1000e15  # 1 ab^-1
    UpsilonToCharged = 0.514
    BplusToDstarLNu = rate.Gamma() * tauBplus
    efficiency = 10e-5
    N = int(eeToUpsilon4S_cross_section * luminosity * 2 * UpsilonToCharged * BplusToDstarLNu * efficiency)
    
    mc = pd.DataFrame(data=rate.generate_events(N), columns=["w", "cosL", "cosV", "chi", "rate"])

    mc["w_bin"] = pd.cut(mc.w, wbins, labels=False)
    mc["cosL_bin"] = pd.cut(mc.cosL, cosLbins, labels=False)
    mc["cosV_bin"] = pd.cut(mc.cosV, cosVbins, labels=False)
    mc["chi_bin"] = pd.cut(mc.chi, chibins, labels=False)

    mc.to_pickle("test.pkl")


if __name__ == "__main__":
    main()


