from spdc_nn.utils.defaults import COINCIDENCE_RATE, DENSITY_MATRIX, TOMOGRAPHY_MATRIX
from spdc_nn.utils.utils import G1_Normalization
from spdc_nn import RES_DIR
from jax import numpy as np

import os
import shutil
import numpy as onp
import matplotlib.pyplot as plt



def save_results(
        run_name,
        observable_vec,
        observables,
        projection_coincidence_rate,
        projection_tomography_matrix,
        Signal,
        Idler,
):
    results_dir = os.path.join(RES_DIR, run_name)
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    (coincidence_rate, density_matrix, tomography_matrix) = observables

    if observable_vec[COINCIDENCE_RATE]:
        coincidence_rate = coincidence_rate[0]
        coincidence_rate = coincidence_rate / np.sum(np.abs(coincidence_rate))
        np.save(os.path.join(results_dir, 'coincidence_rate.npy'), coincidence_rate)
        coincidence_rate_plots(
            results_dir,
            coincidence_rate,
            projection_coincidence_rate,
            Signal,
            Idler,
        )

    if observable_vec[DENSITY_MATRIX]:
        density_matrix = density_matrix[0]
        density_matrix = density_matrix / np.trace(np.real(density_matrix))
        np.save(os.path.join(results_dir, 'density_matrix_real.npy'), onp.real(density_matrix))
        np.save(os.path.join(results_dir, 'density_matrix_imag.npy'), onp.imag(density_matrix))
        density_matrix_plots(
            results_dir,
            density_matrix,
        )

    if observable_vec[TOMOGRAPHY_MATRIX]:
        tomography_matrix = tomography_matrix[0]
        tomography_matrix = tomography_matrix / np.sum(np.abs(tomography_matrix))
        np.save(os.path.join(results_dir, 'tomography_matrix.npy'), tomography_matrix)
        tomography_matrix_plots(
            results_dir,
            tomography_matrix,
            projection_tomography_matrix,
            Signal,
            Idler,
        )


def coincidence_rate_plots(
        results_dir,
        coincidence_rate,
        projection_coincidence_rate,
        Signal,
        Idler,
):
    # coincidence_rate = unwrap_kron(coincidence_rate,
    #                                projection_coincidence_rate.projection_n_modes1,
    #                                projection_coincidence_rate.projection_n_modes2)
    coincidence_rate = coincidence_rate[0, :].\
        reshape(projection_coincidence_rate.projection_n_modes2, projection_coincidence_rate.projection_n_modes2)

    # Compute and plot reduced coincidence_rate
    g1_ss_normalization = G1_Normalization(Signal.w)
    g1_ii_normalization = G1_Normalization(Idler.w)
    coincidence_rate_reduced = coincidence_rate * \
                               projection_coincidence_rate.tau / (g1_ii_normalization * g1_ss_normalization)

    # plot coincidence_rate 2d
    plt.imshow(coincidence_rate_reduced)
    plt.xlabel(r'signal mode i')
    plt.ylabel(r'idle mode j')
    plt.colorbar()

    plt.savefig(os.path.join(results_dir, 'coincidence_rate'))
    plt.close()


def tomography_matrix_plots(
        results_dir,
        tomography_matrix,
        projection_tomography_matrix,
        Signal,
        Idler,
):

    # tomography_matrix = unwrap_kron(tomography_matrix,
    #                                 projection_tomography_matrix.projection_n_state1,
    #                                 projection_tomography_matrix.projection_n_state2)

    tomography_matrix = tomography_matrix[0, :].\
        reshape(projection_tomography_matrix.projection_n_state2, projection_tomography_matrix.projection_n_state2)

    # Compute and plot reduced tomography_matrix
    g1_ss_normalization = G1_Normalization(Signal.w)
    g1_ii_normalization = G1_Normalization(Idler.w)

    tomography_matrix_reduced = tomography_matrix * \
                                projection_tomography_matrix.tau / (g1_ii_normalization * g1_ss_normalization)

    # plot tomography_matrix 2d
    plt.imshow(tomography_matrix_reduced)
    plt.xlabel(r'signal mode i')
    plt.ylabel(r'idle mode j')
    plt.colorbar()

    plt.savefig(os.path.join(results_dir, 'tomography_matrix'))
    plt.close()


def density_matrix_plots(
        results_dir,
        density_matrix,
):

    density_matrix_real = onp.real(density_matrix)
    density_matrix_imag = onp.imag(density_matrix)

    plt.imshow(density_matrix_real)
    plt.xlabel(r'signal mode i')
    plt.ylabel(r'idle mode j')
    plt.colorbar()
    plt.savefig(os.path.join(results_dir, 'density_matrix_real'))
    plt.close()

    plt.imshow(density_matrix_imag)
    plt.xlabel(r'signal mode i')
    plt.ylabel(r'idle mode j')
    plt.colorbar()
    plt.savefig(os.path.join(results_dir, 'density_matrix_imag'))
    plt.close()


def unwrap_kron(G, M1, M2):
    '''
    the function takes a Kronicker product of size M1^2 x M2^2 and turns is into an
    M1 x M2 x M1 x M2 tensor. It is used only for illustration and not during the inference
    Parameters
    ----------
    G: the tensor we wish to reshape
    M1: first dimension
    M2: second dimension

    Returns a reshaped tensor with shape (M1, M2, M1, M2)
    -------

    '''

    C = onp.zeros((M1, M2, M1, M2), dtype=onp.float32)

    for i in range(M1):
        for j in range(M2):
            for k in range(M1):
                for l in range(M2):
                    C[i, j, k, l] = G[k + M1 * i, l + M2 * j]
    return C
