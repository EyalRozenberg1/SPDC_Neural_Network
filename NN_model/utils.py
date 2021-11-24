from abc import ABC
import torch
import numpy as np
from typing import Tuple, Dict, Any, List, Union
from utils_extra import HermiteBank, LaguerreBank, TomographyBank
from defaults import QUBIT, QUTRIT


class Projection_coincidence_rate(ABC):
    """
    A class that represents the projective basis for
    calculating the coincidence rate observable of the interaction.
    """

    def __init__(
            self,
            calculate_observable: Tuple[Dict[Any, bool], ...],
            waist_pump0: float,
            signal_wavelength: float,
            crystal_x,
            crystal_y,
            temperature: float,
            ctype,
            polarization: str,
            z: float = 0.,
            projection_basis: str = 'LG',
            max_mode1: int = 1,
            max_mode2: int = 4,
            waist: float = None,
            wavelength:  float = None,
            tau: float = 1e-9,

    ):
        """

        Parameters
        ----------
        calculate_observable: True/False, will the observable be calculated in simulation
        waist_pump0: pump waists at the center of the crystal (initial)
        signal_wavelength: signal wavelength at spdc interaction
        crystal_x: x axis linspace array (transverse)
        crystal_y: y axis linspace array (transverse)
        temperature: interaction temperature
        ctype: refractive index function
        polarization: polarization for calculating effective refractive index
        z: projection longitudinal position
        projection_basis: type of projection basis
                          Can be: LG (Laguerre-Gauss) / HG (Hermite-Gauss)
        max_mode1: Maximum value of first mode of the 2D projection basis
        max_mode2: Maximum value of second mode of the 2D projection basis
        waist: waists of the projection basis functions
        wavelength: wavelength for generating projection basis
        tau: coincidence window [nano sec]
        """

        self.tau = tau

        if waist is None:
            self.waist = np.sqrt(2) * waist_pump0
        else:
            self.waist = waist

        if wavelength is None:
            wavelength = signal_wavelength

        assert projection_basis.lower() in ['lg', 'hg'], 'The projection basis is LG or HG ' \
                                                         'basis functions only'

        self.projection_basis = projection_basis
        self.max_mode1 = max_mode1
        self.max_mode2 = max_mode2

        # number of modes for projection basis
        if projection_basis.lower() == 'lg':
            self.projection_n_modes1 = max_mode1
            self.projection_n_modes2 = 2 * max_mode2 + 1
        else:
            self.projection_n_modes1 = max_mode1
            self.projection_n_modes2 = max_mode2

        # Total number of projection modes
        self.projection_n_modes = self.projection_n_modes1 * self.projection_n_modes2

        refractive_index = ctype(wavelength * 1e6, temperature, polarization)
        [x, y] = np.meshgrid(crystal_x, crystal_y)

        if calculate_observable:
            if projection_basis.lower() == 'lg':
                self.basis_arr, self.basis_str = \
                    LaguerreBank(
                        wavelength,
                        refractive_index,
                        self.waist,
                        self.max_mode1,
                        self.max_mode2,
                        x, y, z)
            else:
                self.basis_arr, self.basis_str = \
                    HermiteBank(
                        wavelength,
                        refractive_index,
                        self.waist,
                        self.max_mode1,
                        self.max_mode2,
                        x, y, z)


class Projection_tomography_matrix(ABC):
    """
    A class that represents the projective basis for
    calculating the tomography matrix & density matrix observable of the interaction.
    """

    def __init__(
            self,
            calculate_observable: Tuple[Dict[Any, bool], ...],
            waist_pump0: float,
            signal_wavelength: float,
            crystal_x: np.array,
            crystal_y: np.array,
            temperature: float,
            ctype,
            polarization: str,
            z: float = 0.,
            projection_basis: str = 'LG',
            max_mode1: int = 1,
            max_mode2: int = 1,
            waist: float = None,
            wavelength:  float = None,
            tau: float = 1e-9,
            relative_phase: List[Union[Union[int, float], Any]] = None,
            tomography_quantum_state: str = None,

    ):
        """

        Parameters
        ----------
        calculate_observable: True/False, will the observable be calculated in simulation
        waist_pump0: pump waists at the center of the crystal (initial)
        signal_wavelength: signal wavelength at spdc interaction
        crystal_x: x axis linspace array (transverse)
        crystal_y: y axis linspace array (transverse)
        temperature: interaction temperature
        ctype: refractive index function
        polarization: polarization for calculating effective refractive index
        z: projection longitudinal position
        projection_basis: type of projection basis
                          Can be: LG (Laguerre-Gauss)
        max_mode1: Maximum value of first mode of the 2D projection basis
        max_mode2: Maximum value of second mode of the 2D projection basis
        waist: waists of the projection basis functions
        wavelength: wavelength for generating projection basis
        tau: coincidence window [nano sec]
        relative_phase: The relative phase between the mutually unbiased bases (MUBs) states
        tomography_quantum_state: the current quantum state we calculate it tomography matrix.
                                  currently we support: qubit/qutrit
        """

        self.tau = tau

        if waist is None:
            self.waist = np.sqrt(2) * waist_pump0
        else:
            self.waist = waist

        if wavelength is None:
            wavelength = signal_wavelength

        assert projection_basis.lower() in ['lg'], 'The projection basis is LG ' \
                                                   'basis functions only'

        assert max_mode1 == 1, 'for Tomography projections, max_mode1 must be 1'
        assert max_mode2 == 1, 'for Tomography projections, max_mode2 must be 1'

        self.projection_basis = projection_basis
        self.max_mode1 = max_mode1
        self.max_mode2 = max_mode2

        assert tomography_quantum_state in [QUBIT, QUTRIT], f'quantum state must be {QUBIT} or {QUTRIT}, ' \
                                                            'but received {tomography_quantum_state}'
        self.tomography_quantum_state = tomography_quantum_state
        self.relative_phase = relative_phase

        self.projection_n_state1 = 1
        if self.tomography_quantum_state is QUBIT:
            self.projection_n_state2 = 6
        else:
            self.projection_n_state2 = 15

        refractive_index = ctype(wavelength * 1e6, temperature, polarization)
        [x, y] = np.meshgrid(crystal_x, crystal_y)

        if calculate_observable:
            self.basis_arr, self.basis_str = \
                TomographyBank(
                    wavelength,
                    refractive_index,
                    self.waist,
                    self.max_mode1,
                    self.max_mode2,
                    x, y, z,
                    self.relative_phase,
                    self.tomography_quantum_state
                )


def project(projection_basis, beam_profile):
    """
    The function projects some state beam_profile onto given projection_basis
    Parameters
    ----------
    projection_basis: array of basis function
    beam_profile: beam profile (2d)

    Returns
    -------

    """
    Nxx2           = beam_profile.shape[1] ** 2
    N              = beam_profile.shape[0]
    Nh             = projection_basis.shape[0]
    projection     = (torch.conj(projection_basis) * beam_profile).reshape(Nh, N, Nxx2).sum(2)
    normalization1 = torch.abs(beam_profile ** 2).reshape(N, Nxx2).sum(1)
    normalization2 = torch.abs(projection_basis ** 2).reshape(Nh, Nxx2).sum(1)
    projection     = projection / torch.sqrt(normalization1[None, :] * normalization2[:, None])
    return projection


def decompose(beam_profile, projection_basis_arr):
    """
    Decompose a given beam profile into modes defined in the dictionary
    Parameters
    ----------
    beam_profile: beam profile (2d)
    projection_basis_arr: array of basis function

    Returns: beam profile as a decomposition of basis functions
    -------

    """
    projection = project(projection_basis_arr[:, None], beam_profile)
    return projection.T


def fix_power(decomposed_profile, beam_profile):
    """
    Normalize power and ignore higher modes
    Parameters
    ----------
    decomposed_profile: the decomposed beam profile
    beam_profile: the original beam profile

    Returns a normalized decomposed profile
    -------

    """
    scale = torch.sqrt(
        torch.sum(beam_profile * torch.conj(beam_profile), (1, 2))) / torch.sqrt(
        torch.sum(decomposed_profile * torch.conj(decomposed_profile), (1, 2)))

    return decomposed_profile * scale[:, None, None]


def kron(a, b):
    """
    Calculates the kronecker product between two 2d tensors
    Parameters
    ----------
    a, b: 2d tensors
    multiple_devices: (True/False) whether multiple devices are used

    Returns the kronecker product
    -------

    """
    return (a[:, :, None, :, None] * b[:, None, :, None, :]).sum(0)


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
    G1_ss        = kron(np.conj(a), a, ) / N
    G1_ii        = kron(np.conj(b), b, ) / N
    G1_si        = kron(np.conj(b), a, ) / N
    G1_si_dagger = kron(np.conj(a), b, ) / N
    Q_si         = kron(c, a, ) / N
    Q_si_dagger  = kron(np.conj(a), np.conj(c), ) / N

    return G1_ss, G1_ii, G1_si, G1_si_dagger, Q_si, Q_si_dagger


def projection_matrix_calc(G1_ss, G1_ii, G1_si, G1_si_dagger, Q_si, Q_si_dagger):
    """

    Parameters
    ----------
    G1_ss, G1_ii, G1_si, G1_si_dagger, Q_si, Q_si_dagger: the projective matrices
    Returns the 2nd order projective matrix
    -------

    """
    return (G1_ii * G1_ss + Q_si_dagger * Q_si + G1_si_dagger * G1_si).real

