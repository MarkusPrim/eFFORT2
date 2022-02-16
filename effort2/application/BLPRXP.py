import uncertainties
from uncertainties import unumpy as unp
import matplotlib.pyplot as plt
import numpy as np

from effort2.formfactors.BLPRXP import BToDBLPRXP, BToDStarBLPRXP
from effort2.rates.BtoP_MassEffects import BtoP
from effort2.rates.BtoV_MassEffects import BtoV


def main():
    mtau = 1.77682

    pars = parse_fit_result()

    # Remove uncertainties and use pure floats (for profiling)
    # pars = {key: unp.nominal_values(value) for key, value in pars.items()}

    FF_BToD     = BToDBLPRXP(**pars)
    FF_BToD_tau = BToDBLPRXP(**pars, m_L=mtau)
    FF_BToDStar     = BToDStarBLPRXP(**pars)
    FF_BToDStar_tau = BToDStarBLPRXP(**pars, m_L=mtau)

    BToD     = BtoP(FF_BToD, pars["Vcb"])
    BToD_tau = BtoP(FF_BToD_tau, pars["Vcb"])

    BToDStar     = BtoV(FF_BToDStar, pars["Vcb"])
    BToDStar_tau = BtoV(FF_BToDStar_tau, pars["Vcb"])

    wrD    = np.linspace(*FF_BToD.kinematics.w_range_numerical_stable)
    wrDtau = np.linspace(*FF_BToD_tau.kinematics.w_range_numerical_stable)

    wrDs    = np.linspace(*FF_BToDStar.kinematics.w_range_numerical_stable)
    wrDstau = np.linspace(*FF_BToDStar_tau.kinematics.w_range_numerical_stable)


    RD     = BToD_tau.Gamma() / BToD.Gamma()
    RDStar = BToDStar_tau.Gamma() / BToDStar.Gamma()
    corr_RD_RDStar = uncertainties.correlation_matrix([RD, RDStar])[0, 1]

    print(f"""
    $R(D)$ = {RD:.3f}
    $R(D^*)$ = {RDStar:.3f}
    # $\\rho$ = {corr_RD_RDStar:.3f}
    """)


    rateD = np.array([BToD.dGamma_dw(w_) for w_ in wrD])
    rateDtau = np.array([BToD_tau.dGamma_dw(w_) for w_ in wrDtau])

    plot_errorband(wrD, rateD, label=r'$B\to D \ell \nu$')
    plot_errorband(wrDtau, rateDtau, label=r'$B\to D \tau \nu$')
    plt.xlabel('$w$ [1]')
    plt.ylabel('$d\\Gamma/dw$')
    plt.legend()
    plt.tight_layout()
    plt.savefig("BLPRXP_BToD_rate.pdf")
    plt.close()

    del rateD
    del rateDtau

    rateDs = np.array([BToDStar.dGamma_dw(w_) for w_ in wrDs])
    rateDstau = np.array([BToDStar_tau.dGamma_dw(w_) for w_ in wrDstau])

    plot_errorband(wrDs, rateDs, label=r'$B\to D^* \ell \nu$')
    plot_errorband(wrDstau, rateDstau, label=r'$B\to D^* \tau \nu$')
    plt.xlabel('$w$ [1]')
    plt.ylabel('$d\\Gamma/dw$')
    plt.legend()
    plt.tight_layout()
    plt.savefig("BLPRXP_BToDStar_rate.pdf")
    plt.close()

    del rateDs
    del rateDstau



def parse_fit_result():
    central =  np.array([0.038236, 1.100017, 82.442279, -0.057684, 0.000716, 0.032952, 0.267054, -0.117059, 4.709993, 3.399996, 0.208287, 0.008367, 0.401723, 0.129994 ]) 
    errors = np.array([0.000482, 0.001000, 6.638924, 0.019885, 0.019662, 0.018954, 0.027606, 0.176217, 0.049999, 0.049999, 0.727813, 0.264900, 0.295117, 0.020000 ])
    cor = np.array([[1.000000, 0.009051, 0.229939, 0.051798, 0.130253, -0.161319, 0.091392, -0.032673, 0.000000, -0.000000, 0.021283, 0.123246, 0.099580, -0.000000 ],[0.009051, 1.000000, -0.002223, -0.002156, -0.003740, 0.006723, -0.000682, 0.009519, 0.000000, 0.000000, 0.000238, -0.000113, -0.011944, 0.000000 ],[0.229939, -0.002223, 1.000000, -0.085720, 0.244750, 0.267494, 0.144089, 0.150159, 0.000000, -0.000000, -0.208916, -0.230776, -0.088191, -0.000000 ],[0.051798, -0.002156, -0.085720, 1.000000, -0.017820, 0.036327, -0.002999, 0.051180, 0.000000, -0.000000, -0.147005, 0.001209, -0.066755, -0.000000 ],[0.130253, -0.003740, 0.244750, -0.017820, 1.000000, 0.055347, -0.001361, 0.063582, 0.000000, -0.000000, -0.153584, -0.115402, -0.085245, -0.000000 ],[-0.161319, 0.006723, 0.267494, 0.036327, 0.055347, 1.000000, 0.009328, -0.161007, 0.000000, 0.000000, 0.004281, -0.424823, 0.209663, 0.000000 ],[0.091392, -0.000682, 0.144089, -0.002999, -0.001361, 0.009328, 1.000000, 0.039093, 0.000000, 0.000000, -0.090401, -0.065604, 0.108295, -0.000000 ],[-0.032673, 0.009519, 0.150159, 0.051180, 0.063582, -0.161007, 0.039093, 1.000000, 0.000000, -0.000000, -0.903948, -0.719361, -0.867228, -0.000000 ],[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000 ],[-0.000000, 0.000000, -0.000000, -0.000000, -0.000000, 0.000000, 0.000000, -0.000000, -0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000 ],[0.021283, 0.000238, -0.208916, -0.147005, -0.153584, 0.004281, -0.090401, -0.903948, -0.000000, 0.000000, 1.000000, 0.874460, 0.902600, 0.000000 ],[0.123246, -0.000113, -0.230776, 0.001209, -0.115402, -0.424823, -0.065604, -0.719361, -0.000000, 0.000000, 0.874460, 1.000000, 0.713234, 0.000000 ],[0.099580, -0.011944, -0.088191, -0.066755, -0.085245, 0.209663, 0.108295, -0.867228, -0.000000, 0.000000, 0.902600, 0.713234, 1.000000, 0.000000 ],[-0.000000, 0.000000, -0.000000, -0.000000, -0.000000, 0.000000, -0.000000, -0.000000, -0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000 ]])

    cov = np.outer(errors, errors) * cor
    pars = {}
    (pars["Vcb"],pars["RhoSq"],pars["Cur"],pars["chi21"],pars["chi2p"],pars["chi3p"],pars["eta1"],pars["etap"],pars["mb1S"],pars["dmbc"],pars["beta21"],pars["beta3p"],pars["phi1p"],pars["la2"]) = uncertainties.correlated_values(central,cov)

    return pars


def plot_errorband(x, y, label=None, color=None, ls='solid', alpha=0.3):
    p = plt.plot(
        x, unp.nominal_values(y), 
        color=color, ls=ls, label=label
    )
    plt.fill_between(
        x, unp.nominal_values(y) - unp.std_devs(y), unp.nominal_values(y) + unp.std_devs(y),
        color=p[0].get_color(), ls="-", alpha=alpha, )


def profile(f):
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        return_value = f()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    stats.dump_stats(filename="BLPRXP.prof")
    return return_value


if __name__ == "__main__":
    main()
