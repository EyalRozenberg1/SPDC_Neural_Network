import random
import os
import shutil
import time
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List, Union
from spdc_nn import LOGS_DIR
from spdc_nn.utils.utils import Beam
from spdc_nn.utils.defaults import COINCIDENCE_RATE, DENSITY_MATRIX, TOMOGRAPHY_MATRIX
from spdc_nn.utils.defaults import REAL, IMAG
from spdc_nn.data.interaction import Interaction
from spdc_nn.forward.utils import Projection_coincidence_rate, Projection_tomography_matrix
from spdc_nn.forward.results_and_stats_utils import save_results
from spdc_nn.forward.base_forward import BaseForward
import numpy as np


def run_forward(
        run_name: str,
        seed: int = 42,
        CUDA_VISIBLE_DEVICES: str = None,
        JAX_ENABLE_X64: str = 'True',
        minimal_GPU_memory: bool = False,
        N: int = 4000,
        observable_vec: Tuple[Dict[Any, bool]] = None,
        tau: float = 1e-9,
        pump_basis: str = 'LG',
        pump_max_mode1: int = 5,
        pump_max_mode2: int = 3,
        pump_coefficient: str = 'random',
        custom_pump_coefficient: Dict[str, Dict[int, int]] = None,
        pump_coefficient_path: str = None,
        pump_waist: str = 'waist_pump0',
        pump_waists_path: str = None,
        crystal_basis: str = 'LG',
        crystal_max_mode1: int = 5,
        crystal_max_mode2: int = 3,
        crystal_coefficient: str = 'random',
        custom_crystal_coefficient: Dict[str, Dict[int, int]] = None,
        crystal_coefficient_path: str = None,
        crystal_waist: str = 'r_scale0',
        crystal_waists_path: str = None,
        lam_pump: float = 405e-9,
        crystal_str: str = 'ktp',
        power_pump: float = 1e-3,
        waist_pump0: float = 40e-6,
        r_scale0: float = 40e-6,
        dx: float = 4e-6,
        dy: float = 4e-6,
        dz: float = 10e-6,
        maxX: float = 120e-6,
        maxY: float = 120e-6,
        maxZ: float = 1e-3,
        R: float = 0.1,
        Temperature: float = 50,
        pump_polarization: str = 'y',
        signal_polarization: str = 'y',
        idler_polarization: str = 'z',
        dk_offset: float = 1.,
        power_signal: float = 1.,
        power_idler: float = 1.,
        coincidence_projection_basis: str = 'LG',
        coincidence_projection_max_mode1: int = 1,
        coincidence_projection_max_mode2: int = 4,
        coincidence_projection_waist: float = None,
        coincidence_projection_wavelength: float = None,
        coincidence_projection_polarization: str = 'y',
        coincidence_projection_z: float = 0.,
        tomography_projection_basis: str = 'LG',
        tomography_projection_max_mode1: int = 1,
        tomography_projection_max_mode2: int = 1,
        tomography_projection_waist: float = None,
        tomography_projection_wavelength: float = None,
        tomography_projection_polarization: str = 'y',
        tomography_projection_z: float = 0.,
        tomography_relative_phase: List[Union[Union[int, float], Any]] = None,
        tomography_quantum_state: str = 'qubit',
        key = None

):
    """
    This function is the main function for running SPDC project

    Parameters
    ----------
    run_name: selected name (will be used for naming the folder)
    seed: initial seed for random functions
    CUDA_VISIBLE_DEVICES: visible gpu devices to be used
    JAX_ENABLE_X64: if True, use double-precision numbers (enabling 64bit mode)
    minimal_GPU_memory: This makes JAX allocate exactly what is needed on demand, and deallocate memory that is no
                        longer needed (note that this is the only configuration that will deallocate GPU memory,
                        instead of reusing it). This is very slow, so is not recommended for general use,
                        but may be useful for running with the minimal possible GPU memory footprint
                        or debugging OOM failures.
    N: size of vacuum states in inference method
    observable_vec: if an observable in the dictionary is True,
                        the method will infer the observable along the process
    pump_basis: Pump's construction basis method
                Can be: LG (Laguerre-Gauss) / HG (Hermite-Gauss)
    pump_max_mode1: Maximum value of first mode of the 2D pump basis
    pump_max_mode2: Maximum value of second mode of the 2D pump basis
    pump_coefficient: defines the initial distribution of coefficient-amplitudes for pump basis function
                              can be: uniform- uniform distribution
                                      random- uniform distribution
                                      custom- as defined at custom_pump_coefficient
                                      load- will be loaded from np.arrays defined under path: pump_coefficient_path
                                            with names: PumpCoeffs_real.npy, PumpCoeffs_imag.npy
    pump_coefficient_path: path for loading waists for pump basis function
    custom_pump_coefficient: (dictionary) used only if pump_coefficient=='custom'
                             {'real': {indexes:coeffs}, 'imag': {indexes:coeffs}}.
    pump_waist: defines the initial values of waists for pump basis function
                        can be: waist_pump0- will be set according to waist_pump0
                                load- will be loaded from np.arrays defined under path: pump_waists_path
                                with name: PumpWaistCoeffs.npy
    pump_waists_path: path for loading coefficient-amplitudes for pump basis function
    crystal_basis: Crystal's construction basis method
                   Can be:
                   None / FT (Fourier-Taylor) / FB (Fourier-Bessel) / LG (Laguerre-Gauss) / HG (Hermite-Gauss)
                   - if None, the crystal will contain NO hologram
    crystal_max_mode1: Maximum value of first mode of the 2D crystal basis
    crystal_max_mode2: Maximum value of second mode of the 2D crystal basis
    crystal_coefficient: defines the initial distribution of coefficient-amplitudes for crystal basis function
                                 can be: uniform- uniform distribution
                                  random- uniform distribution
                                  custom- as defined at custom_crystal_coefficient
                                  load- will be loaded from np.arrays defined under path: crystal_coefficient_path
                                        with names: CrystalCoeffs_real.npy, CrystalCoeffs_imag.npy
    crystal_coefficient_path: path for loading coefficient-amplitudes for crystal basis function
    custom_crystal_coefficient: (dictionary) used only if crystal_coefficient=='custom'
                             {'real': {indexes:coeffs}, 'imag': {indexes:coeffs}}.
    crystal_waist: defines the initial values of waists for crystal basis function
                           can be: r_scale0- will be set according to r_scale0
                                   load- will be loaded from np.arrays defined under path: crystal_waists_path
                                         with name: CrystalWaistCoeffs.npy
    crystal_waists_path: path for loading waists for crystal basis function
    lam_pump: Pump wavelength
    crystal_str: Crystal type. Can be: KTP or MgCLN
    power_pump: Pump power [watt]
    waist_pump0: waists of the pump basis functions.
                 -- If None, waist_pump0 = sqrt(maxZ / self.pump_k)
    r_scale0: effective waists of the crystal basis functions.
              -- If None, r_scale0 = waist_pump0
    dx: transverse resolution in x [m]
    dy: transverse resolution in y [m]
    dz: longitudinal resolution in z [m]
    maxX: Transverse cross-sectional size from the center of the crystal in x [m]
    maxY: Transverse cross-sectional size from the center of the crystal in y [m]
    maxZ: Crystal's length in z [m]
    R: distance to far-field screen [m]
    Temperature: crystal's temperature [Celsius Degrees]
    pump_polarization: Polarization of the pump beam
    signal_polarization: Polarization of the signal beam
    idler_polarization: Polarization of the idler beam
    dk_offset: delta_k offset
    power_signal: Signal power [watt]
    power_idler: Idler power [watt]

    coincidence_projection_basis: represents the projective basis for calculating the coincidence rate observable
                                  of the interaction. Can be: LG (Laguerre-Gauss) / HG (Hermite-Gauss)
    coincidence_projection_max_mode1: Maximum value of first mode of the 2D projection basis for coincidence rate
    coincidence_projection_max_mode2: Maximum value of second mode of the 2D projection basis for coincidence rate
    coincidence_projection_waist: waists of the projection basis functions of coincidence rate.
                                  if None, np.sqrt(2) * waist_pump0 is used
    coincidence_projection_wavelength: wavelength for generating projection basis of coincidence rate.
                                       if None, the signal wavelength is used
    coincidence_projection_polarization: polarization for calculating effective refractive index
    coincidence_projection_z: projection longitudinal position
    tomography_projection_basis: represents the projective basis for calculating the tomography matrix & density matrix
                                    observables of the interaction. Can be: LG (Laguerre-Gauss) / HG (Hermite-Gauss)
    tomography_projection_max_mode1: Maximum value of first mode of the 2D projection basis for tomography matrix &
                                        density matrix
    tomography_projection_max_mode2: Maximum value of second mode of the 2D projection basis for tomography matrix &
                                        density matrix
    tomography_projection_waist: waists of the projection basis functions of tomography matrix & density matrix
                                  if None, np.sqrt(2) * waist_pump0 is used
    tomography_projection_wavelength: wavelength for generating projection basis of tomography matrix & density matrix.
                                       if None, the signal wavelength is used
    tomography_projection_polarization: polarization for calculating effective refractive index
    tomography_projection_z: projection longitudinal position
    tomography_relative_phase: The relative phase between the mutually unbiased bases (MUBs) states
    tomography_quantum_state: the current quantum state we calculate it tomography matrix.
                                currently we support: qubit/qutrit
    tau: coincidence window [nano sec]
    -------

    """
    run_name = f'i_{run_name}_{str(datetime.today()).split()[0]}'

    now = datetime.now()
    date_and_time = now.strftime("%d/%m/%Y %H:%M:%S")
    print("date and time =", date_and_time)

    if CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    if JAX_ENABLE_X64:
        os.environ["JAX_ENABLE_X64"] = JAX_ENABLE_X64

    if minimal_GPU_memory:
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = 'platform'

    import jax
    from jax.lib import xla_bridge

    if not seed:
        seed = random.randint(0, 2 ** 31)
    if key is None:
        key = jax.random.PRNGKey(seed)

    n_devices = xla_bridge.device_count()
    print(f'Number of GPU devices: {n_devices} \n')

    assert N % n_devices == 0, "The number of examples should be " \
                               "divisible by the number of devices"
    N_device = int(N / n_devices)

    specs = {
        'experiment name': run_name,
        'seed': seed,
        'date and time': date_and_time,
        'number of gpu devices': n_devices,
        'JAX_ENABLE_X64': JAX_ENABLE_X64,
    }
    specs.update({'----- Simulation Parameters': '----- '})
    specs.update(simulation_params)
    specs.update({'----- Interaction Parameters': '----- '})
    specs.update(interaction_params)
    specs.update({'----- Projection Parameters': '----- '})
    specs.update(projection_params)

    logs_dir = os.path.join(LOGS_DIR, run_name)
    if os.path.exists(logs_dir):
        shutil.rmtree(logs_dir)
    os.makedirs(logs_dir, exist_ok=True)

    _, interaction_key = jax.random.split(key)
    interaction = Interaction(
        pump_basis=pump_basis,
        pump_max_mode1=pump_max_mode1,
        pump_max_mode2=pump_max_mode2,
        pump_coefficient=pump_coefficient,
        custom_pump_coefficient=custom_pump_coefficient,
        pump_coefficient_path=pump_coefficient_path,
        pump_waist=pump_waist,
        pump_waists_path=pump_waists_path,
        crystal_basis=crystal_basis,
        crystal_max_mode1=crystal_max_mode1,
        crystal_max_mode2=crystal_max_mode2,
        crystal_coefficient=crystal_coefficient,
        custom_crystal_coefficient=custom_crystal_coefficient,
        crystal_coefficient_path=crystal_coefficient_path,
        crystal_waist=crystal_waist,
        crystal_waists_path=crystal_waists_path,
        lam_pump=lam_pump,
        crystal_str=crystal_str,
        power_pump=power_pump,
        waist_pump0=waist_pump0,
        r_scale0=r_scale0,
        dx=dx,
        dy=dy,
        dz=dz,
        maxX=maxX,
        maxY=maxY,
        maxZ=maxZ,
        R=R,
        Temperature=Temperature,
        pump_polarization=pump_polarization,
        signal_polarization=signal_polarization,
        idler_polarization=idler_polarization,
        dk_offset=dk_offset,
        power_signal=power_signal,
        power_idler=power_idler,
        key=interaction_key,
    )

    projection_coincidence_rate = Projection_coincidence_rate(
        calculate_observable=observable_vec[COINCIDENCE_RATE],
        waist_pump0=interaction.waist_pump0,
        signal_wavelength=interaction.lam_signal,
        crystal_x=interaction.x,
        crystal_y=interaction.y,
        temperature=interaction.Temperature,
        ctype=interaction.ctype,
        polarization=coincidence_projection_polarization,
        z=coincidence_projection_z,
        projection_basis=coincidence_projection_basis,
        max_mode1=coincidence_projection_max_mode1,
        max_mode2=coincidence_projection_max_mode2,
        waist=coincidence_projection_waist,
        wavelength=coincidence_projection_wavelength,
        tau=tau
    )

    projection_tomography_matrix = Projection_tomography_matrix(
        calculate_observable=observable_vec[DENSITY_MATRIX] or observable_vec[TOMOGRAPHY_MATRIX],
        waist_pump0=interaction.waist_pump0,
        signal_wavelength=interaction.lam_signal,
        crystal_x=interaction.x,
        crystal_y=interaction.y,
        temperature=interaction.Temperature,
        ctype=interaction.ctype,
        polarization=tomography_projection_polarization,
        z=tomography_projection_z,
        relative_phase=tomography_relative_phase,
        tomography_quantum_state=tomography_quantum_state,
        projection_basis=tomography_projection_basis,
        max_mode1=tomography_projection_max_mode1,
        max_mode2=tomography_projection_max_mode2,
        waist=tomography_projection_waist,
        wavelength=tomography_projection_wavelength,
        tau=tau,
    )

    Pump = Beam(lam=interaction.lam_pump,
                ctype=interaction.ctype,
                polarization=interaction.pump_polarization,
                T=interaction.Temperature,
                power=interaction.power_pump)

    Signal = Beam(lam=interaction.lam_signal,
                  ctype=interaction.ctype,
                  polarization=interaction.signal_polarization,
                  T=interaction.Temperature,
                  power=interaction.power_signal)

    Idler = Beam(lam=interaction.lam_idler,
                 ctype=interaction.ctype,
                 polarization=interaction.idler_polarization,
                 T=interaction.Temperature,
                 power=interaction.power_idler)

    forward = BaseForward(
        key=key,
        N=N,
        N_device=N_device,
        n_devices=n_devices,
        projection_coincidence_rate=projection_coincidence_rate,
        projection_tomography_matrix=projection_tomography_matrix,
        interaction=interaction,
        pump=Pump,
        signal=Signal,
        idler=Idler,
        observable_vec=observable_vec, )

    start_time = time.time()
    observables, signal_kappa, idler_kappa = forward.inference()
    pump_coeffs = np.concatenate(
        (np.expand_dims(np.array(forward.pump_coeffs_real), 0), np.expand_dims(np.array(forward.pump_coeffs_imag), 0)),
        axis=0)
    # without zero for crystal
    # crystal_coeffs = np.concatenate((np.expand_dims(np.array(forward.crystal_coeffs_real), 0),
    #                                  np.expand_dims(np.array(forward.crystal_coeffs_imag), 0)), axis=0)
    crystal_coeffs = np.zeros_like(pump_coeffs)
    pump_crystal = np.concatenate((np.expand_dims(pump_coeffs, 0), np.expand_dims(crystal_coeffs, 0)), axis=0)
    idler_signal_coeffs = np.concatenate((idler_kappa, signal_kappa), axis=1).squeeze()

    total_time = (time.time() - start_time)
    print("inference is done after: %s seconds" % total_time)

    return observables[0][0], idler_signal_coeffs, pump_crystal, interaction.key
    # save_results(
    #     run_name,
    #     observable_vec,
    #     observables,
    #     projection_coincidence_rate,
    #     projection_tomography_matrix,
    #     Signal,
    #     Idler,
    # )

    # specs_file = os.path.join(logs_dir, 'data_specs.txt')
    # with open(specs_file, 'w') as f:
    #     f.write(f"running time: {total_time} sec\n")
    #     for k, v in specs.items():
    #         f.write(f"{k}: {str(v)}\n")


if __name__ == "__main__":
    simulation_params = {
        'N': 100,
        'observable_vec': {
            COINCIDENCE_RATE: True,
            DENSITY_MATRIX: False,
            TOMOGRAPHY_MATRIX: False
        }
    }

    interaction_params = {
        'pump_max_mode1': 1,
        'pump_max_mode2': 3,
        'pump_coefficient': 'random',
        'custom_pump_coefficient': {REAL: {0: 1., 1: 0., 2: 0., 3: 0., 4: 1., 5: 0., 6: 1., 7: 0., 8: 0.},
                                    IMAG: {0: 0., 1: 0., 2: 0.}},
        'pump_coefficient_path': None,
        'pump_waist': 'waist_pump0',
        'pump_waists_path': None,
        'crystal_basis': None,
        'crystal_max_mode1': None,
        'crystal_max_mode2': None,
        'crystal_coefficient': 'random',
        'custom_crystal_coefficient': {REAL: {-1: 1, 0: 1, 1: 1}, IMAG: {-1: 1, 0: 1, 1: 1}},
        'crystal_coefficient_path': None,
        'crystal_waist': 'r_scale0',
        'crystal_waists_path': None,
        'lam_pump': 405e-9,
        'crystal_str': 'ktp',
        'power_pump': 1e-3,
        'waist_pump0': 40e-6,
        'r_scale0': 40e-6,
        'dx': 4e-6,
        'dy': 4e-6,
        'dz': 10e-6,
        'maxX': 160e-6,
        'maxY': 160e-6,
        'maxZ': 1e-3,
    }

    projection_params = {
        'coincidence_projection_basis': 'LG',
        'coincidence_projection_max_mode1': 1,
        'coincidence_projection_max_mode2': 4,
    }

    key = None
    for i in range(1000):
        coincidence_rate, idler_signal_coeffs, pump_crystal, key = run_forward(
        # run_forward(
            run_name=f'test{i}',
            # seed=4,
            seed=42,
            JAX_ENABLE_X64='True',
            minimal_GPU_memory=False,
            CUDA_VISIBLE_DEVICES='0',
            **simulation_params,
            **interaction_params,
            **projection_params,
            key=key
        )
        coincidence_rate = coincidence_rate / np.sum(np.abs(coincidence_rate))
        if i == 0:
            concat_coincidence_rate = coincidence_rate
            concat_idler_signal_coeffs = np.expand_dims(idler_signal_coeffs, 0)
            concat_pump_crystal = np.expand_dims(pump_crystal, 0)
        else:
            concat_coincidence_rate = np.concatenate((concat_coincidence_rate, coincidence_rate), axis=0)
            concat_idler_signal_coeffs = np.concatenate(
                (concat_idler_signal_coeffs, np.expand_dims(idler_signal_coeffs, 0)), axis=0)
            concat_pump_crystal = np.concatenate((concat_pump_crystal, np.expand_dims(pump_crystal, 0)), axis=0)
        print(f'current_run: {i}')

    np.save('coincidence_rate_new_all_zero_except_pump.npy', concat_coincidence_rate)
    np.save('idler_signal_coeffs_new_all_zero_except_pump.npy', concat_idler_signal_coeffs)
    np.save('pump_crystal_coeffs_new_all_zero_except_pump.npy', concat_pump_crystal)
