import torch

from utils import Projection_coincidence_rate, kron, fix_power, decompose, projection_matrix_calc
import numpy as np

import operator

N = 100


def n_KTP_Kato(
        lam: float,
        T: float,
        ax: str,
):
    """
    Refractive index for KTP, based on K. Kato

    Parameters
    ----------
    lam: wavelength (lambda) [um]
    T: Temperature [Celsius Degrees]
    ax: polarization

    Returns
    -------
    n: Refractive index

    """
    assert ax in ['z', 'y'], 'polarization must be either z or y'
    dT = (T - 20)
    if ax == "z":
        n_no_T_dep = np.sqrt(4.59423 + 0.06206 / (lam ** 2 - 0.04763) + 110.80672 / (lam ** 2 - 86.12171))
        dn = (0.9221 / lam ** 3 - 2.9220 / lam ** 2 + 3.6677 / lam - 0.1897) * 1e-5 * dT
    if ax == "y":
        n_no_T_dep = np.sqrt(3.45018 + 0.04341 / (lam ** 2 - 0.04597) + 16.98825 / (lam ** 2 - 39.43799))
        dn = (0.1997 / lam ** 3 - 0.4063 / lam ** 2 + 0.5154 / lam + 0.5425) * 1e-5 * dT
    n = n_no_T_dep + dn
    return n


def projections_init():
    g_cr = torch.zeros((1,
                     1,
                     9,
                     9), dtype=torch.complex64)

    coincidence_rate_projections_init = (torch.zeros((1, 1, 9, 9), dtype=torch.complex64).cuda(),
                                         torch.zeros_like(g_cr).cuda(),
                                         torch.zeros_like(g_cr).cuda(),
                                         torch.zeros_like(g_cr).cuda(),
                                         torch.zeros_like(g_cr).cuda(),
                                         torch.zeros_like(g_cr).cuda())

    return coincidence_rate_projections_init, None


def decompose1(
        signal_out,
        idler_out,
        idler_vac,
        basis_arr,
        projection_n_1,
        projection_n_2
):
    signal_beam_decompose = decompose(
        signal_out,
        basis_arr
    ).reshape(
        N,
        projection_n_1,
        projection_n_2)

    idler_beam_decompose = decompose(
        idler_out,
        basis_arr
    ).reshape(
        N,
        projection_n_1,
        projection_n_2)

    idler_vac_decompose = decompose(
        idler_vac,
        basis_arr
    ).reshape(
        N,
        projection_n_1,
        projection_n_2)

    # say there are no higher modes by normalizing the power
    signal_beam_decompose = fix_power(signal_beam_decompose, signal_out)
    idler_beam_decompose = fix_power(idler_beam_decompose, idler_out)
    idler_vac_decompose = fix_power(idler_vac_decompose, idler_vac)

    return signal_beam_decompose, idler_beam_decompose, idler_vac_decompose


def projection_matrices_calc(a, b, c, N, multiple_devices: bool = False):
    """

    Parameters
    ----------
    a, b, c: the interacting fields
    N: Total number of interacting vacuum state elements
    multiple_devices: (True/False) whether multiple devices are used

    Returns the projective matrices
    -------

    """
    G1_ss = kron(torch.conj(a), a, ) / N
    G1_ii = kron(torch.conj(b), b, ) / N
    G1_si = kron(torch.conj(b), a, ) / N
    G1_si_dagger = kron(torch.conj(a), b, ) / N
    Q_si = kron(c, a) / N
    Q_si_dagger = kron(torch.conj(a), torch.conj(c)) / N

    return G1_ss, G1_ii, G1_si, G1_si_dagger, Q_si, Q_si_dagger


projection_coincidence_rate = Projection_coincidence_rate(
    calculate_observable=True,
    waist_pump0=4e-5,
    signal_wavelength=810e-9,
    crystal_x=np.arange(-160e-6, 160e-6, 4e-6),
    crystal_y=np.arange(-160e-6, 160e-6, 4e-6),
    temperature=50,
    ctype=n_KTP_Kato,
    polarization='y',
    z=0.0,
    projection_basis='LG',
    max_mode1=1,
    max_mode2=4,
    waist=None,
    wavelength=None,
    tau=1e-9
)
projection_coincidence_rate.basis_arr = torch.tensor(projection_coincidence_rate.basis_arr).cuda()


def decompose_and_get_projections(
        signal_out,
        idler_out,
        idler_vac,
        basis_arr,
        projection_n_1,
        projection_n_2
):
    """
    The function decompose the interacting fields onto selected basis array, and calculates first order
        correlation functions according to  https://doi.org/10.1002/lpor.201900321

    Parameters
    ----------
    signal_out
    idler_out
    idler_vac
    basis_arr
    projection_n_1
    projection_n_2

    Returns
    -------

    """

    signal_beam_decompose, idler_beam_decompose, idler_vac_decompose = \
        decompose1(
            signal_out,
            idler_out,
            idler_vac,
            basis_arr,
            projection_n_1,
            projection_n_2
        )

    G1_ss, G1_ii, G1_si, G1_si_dagger, Q_si, Q_si_dagger = projection_matrices_calc(
        signal_beam_decompose,
        idler_beam_decompose,
        idler_vac_decompose,
        N
    )

    return G1_ss, G1_ii, G1_si, G1_si_dagger, Q_si, Q_si_dagger


def calc_observables(idler_out, signal_out, idler_vac):
    coincidence_rate_projections, _ = projections_init()

    coincidence_rate_projections_ = decompose_and_get_projections(
        signal_out,
        idler_out,
        idler_vac,
        projection_coincidence_rate.basis_arr,
        1,
        9
    )

    coincidence_rate_projections = tuple(map(operator.add, coincidence_rate_projections, coincidence_rate_projections_))

    coincidence_rate = projection_matrix_calc(
        *coincidence_rate_projections
    ).reshape(
        1 ** 2,
        9 ** 2)
    return coincidence_rate
