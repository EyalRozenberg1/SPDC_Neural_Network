import jax.random as random
from abc import ABC
from typing import Dict, Optional, Tuple, Any
from jax import pmap
from jax import numpy as np
from jax.lax import stop_gradient
from spdc_nn.utils.utils import Crystal_hologram, Beam_profile
from spdc_nn.models.spdc_model import SPDCmodel
from spdc_nn.utils.defaults import COINCIDENCE_RATE, DENSITY_MATRIX, TOMOGRAPHY_MATRIX


class BaseForward(ABC):
    """
    A class abstracting the various tasks of running the forward model.
    Provides methods at multiple levels of granularity
    """
    def __init__(
            self,
            key: np.array,
            N: int,
            N_device: int,
            n_devices: int,
            projection_coincidence_rate,
            projection_tomography_matrix,
            interaction,
            pump,
            signal,
            idler,
            observable_vec: Optional[Tuple[Dict[Any, bool]]],
    ):

        self.key       = key
        self.n_devices = n_devices
        self.N         = N
        self.N_device  = N_device
        self.delta_k   = pump.k - signal.k - idler.k  # phase mismatch
        self.Nx        = interaction.Nx
        self.Ny        = interaction.Ny
        self.DeltaZ    = - interaction.maxZ / 2  # DeltaZ: longitudinal middle of the crystal (with negative sign).
                                              # To propagate generated fields back to the middle of the crystal
        self.poling_period = interaction.dk_offset * self.delta_k

        self.projection_coincidence_rate  = projection_coincidence_rate
        self.projection_tomography_matrix = projection_tomography_matrix

        assert list(observable_vec.keys()) == [COINCIDENCE_RATE,
                                               DENSITY_MATRIX,
                                               TOMOGRAPHY_MATRIX], 'observable_vec must only contain ' \
                                                                   'the keys [coincidence_rate,' \
                                                                   'density_matrix, tomography_matrix]'

        self.coincidence_rate_observable = observable_vec[COINCIDENCE_RATE]
        self.density_matrix_observable = observable_vec[DENSITY_MATRIX]
        self.tomography_matrix_observable = observable_vec[TOMOGRAPHY_MATRIX]

        # Initialize pump and crystal coefficients
        self.pump_coeffs_real, \
        self.pump_coeffs_imag = interaction.pump_coefficients()
        self.waist_pump       = interaction.pump_waists()

        self.crystal_coeffs_real,\
        self.crystal_coeffs_imag = interaction.crystal_coefficients()
        self.r_scale             = interaction.crystal_waists()

        self.model_parameters = pmap(lambda x: (
                                                self.pump_coeffs_real,
                                                self.pump_coeffs_imag,
                                                self.waist_pump,
                                                self.crystal_coeffs_real,
                                                self.crystal_coeffs_imag,
                                                self.r_scale
        ))(np.arange(self.n_devices))

        print(f"Interaction length [m]: {interaction.maxZ} \n")
        print(f"Pump   beam  basis  coefficients: \n {self.pump_coeffs_real + 1j * self.pump_coeffs_imag}\n")
        print(f"Pump basis functions waists [um]: \n {self.waist_pump * 10}\n")

        if interaction.crystal_basis:
            print(f"3D hologram  basis  coefficients: \n {self.crystal_coeffs_real + 1j * self.crystal_coeffs_imag}\n")
            print("3D hologram basis functions-"
                  f"effective  waists (r_scale) [um]: \n {self.r_scale * 10}\n")
            self.crystal_hologram = Crystal_hologram(self.crystal_coeffs_real,
                                                     self.crystal_coeffs_imag,
                                                     self.r_scale,
                                                     interaction.x,
                                                     interaction.y,
                                                     interaction.crystal_max_mode1,
                                                     interaction.crystal_max_mode2,
                                                     interaction.crystal_basis,
                                                     signal.lam,
                                                     signal.n,)
        else:
            self.crystal_hologram = None

        self.pump_structure = Beam_profile(self.pump_coeffs_real,
                                           self.pump_coeffs_imag,
                                           self.waist_pump,
                                           interaction.power_pump,
                                           interaction.x,
                                           interaction.y,
                                           interaction.dx,
                                           interaction.dy,
                                           interaction.pump_max_mode1,
                                           interaction.pump_max_mode2,
                                           interaction.pump_basis,
                                           interaction.lam_pump,
                                           pump.n,)

        self.model = SPDCmodel(pump,
                               signal=signal,
                               idler=idler,
                               projection_coincidence_rate=projection_coincidence_rate,
                               projection_tomography_matrix=projection_tomography_matrix,
                               interaction=interaction,
                               pump_structure=self.pump_structure,
                               crystal_hologram=self.crystal_hologram,
                               poling_period=self.poling_period,
                               DeltaZ=self.DeltaZ,
                               coincidence_rate_observable=self.coincidence_rate_observable,
                               density_matrix_observable=self.density_matrix_observable,
                               tomography_matrix_observable=self.tomography_matrix_observable,)

    def inference(self):
        self.model.N           = self.N
        self.model.N_device    = self.N_device

        # seed vacuum samples for each gpu
        self.key, subkey = random.split(self.key)
        keys = random.split(subkey, self.n_devices)
        observables = pmap(self.model.forward, axis_name='device')(stop_gradient(self.model_parameters), keys)

        return observables

