from utils_extra import G1_Normalization
from enangelment import hyperparams
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

c = 2.99792458e8


def save_results(
        run_name,
        coincidence_rate
):
    results_dir = os.path.join('/home/barak/Documents/Project/project_on_pycharm/', run_name)
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # coincidence_rate = coincidence_rate[0]
    coincidence_rate = coincidence_rate / np.sum(np.abs(coincidence_rate.cpu().numpy()))
    # plots aren't needed for data mining purpose
    coincidence_rate_plots(
        results_dir,
        coincidence_rate
    )


def coincidence_rate_plots(
        results_dir,
        coincidence_rate
):
    coincidence_rate = unwrap_kron(coincidence_rate,
                                   1,
                                   9)

    # Compute and plot reduced coincidence_rate
    g1_ss_normalization = G1_Normalization(2 * np.pi * c / hyperparams['lambda_signal'])
    g1_ii_normalization = G1_Normalization(2 * np.pi * c / hyperparams['lambda_idler'])
    coincidence_rate_reduced = coincidence_rate[0, :, 0, :] * \
                               hyperparams['tau'] / (g1_ii_normalization * g1_ss_normalization)

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
    tomography_matrix = unwrap_kron(tomography_matrix,
                                    projection_tomography_matrix.projection_n_state1,
                                    projection_tomography_matrix.projection_n_state2)

    # Compute and plot reduced tomography_matrix
    g1_ss_normalization = G1_Normalization(Signal.w)
    g1_ii_normalization = G1_Normalization(Idler.w)

    tomography_matrix_reduced = tomography_matrix[0, :, 0, :] * \
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
    density_matrix_real = np.real(density_matrix)
    density_matrix_imag = np.imag(density_matrix)

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

    C = np.zeros((M1, M2, M1, M2), dtype=np.float32)

    for i in range(M1):
        for j in range(M2):
            for k in range(M1):
                for l in range(M2):
                    C[i, j, k, l] = G[k + M1 * i, l + M2 * j]
    return C
