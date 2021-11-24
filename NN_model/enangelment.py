import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset
import torch.utils.data as data
from coincidence_calculation import calc_observables

device = 'cuda:0' if True else 'cpu'
SFG_idler_wavelength = lambda lambda_p, lambda_s: lambda_p * lambda_s / (lambda_s - lambda_p)
torch.set_default_dtype(torch.float32)
torch.set_printoptions(precision=8)


class Sine(nn.Module):
    def __init__(self, frequency: float = 30):
        super().__init__()
        self.frequency = frequency

    def forward(self, x):
        return torch.sin(self.frequency * x)


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


hyperparams = {'kp': (2 * np.pi * n_KTP_Kato(405e-9 * 1e6, T=50, ax='y') / 405e-9),
               'ks': (2 * np.pi * n_KTP_Kato(810e-9 * 1e6, T=50, ax='y') / 810e-9),
               'ki': (2 * np.pi * n_KTP_Kato(810e-9 * 1e6, T=50, ax='z') / 1064e-9),
               'chi2': 16.9e-12,
               'lambda_pump': 405e-9,  #
               'tau': 1e-9,
               'lambda_signal': 2 * 405e-9,
               'lambda_idler': SFG_idler_wavelength(405e-9, 2 * 405e-9),  # Harmonic mean of the lam_pump and lam_signal
               'sample_space': 81 * 81,  # sample space size. 4000
               'basis_coefficients_pump': 14,
               # number of basis coefficients - pump. (3(mod2)*2 + 1)*5(mod1) - 2 -real,imag
               'basis_coefficients_crystal': 0,  # number of basis coefficients - crystal.
               'z_0': 4.90001694e-04,  # z is constant for now.
               'projection_basis_size': 9 ** 2,  # for projection p = 1 ,l = 4
               'N': 100}


def get_xy_couples(maxX, maxY, dx, dy):
    x_axis = np.arange(-maxX, maxX, dx)
    y_axis = np.arange(-maxY, maxY, dy)  # y axis, length 2*MaxY  (transverse)
    X, Y = np.meshgrid(x_axis, y_axis, indexing='ij')
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)
    xy_couples = torch.cat((X.unsqueeze(2), Y.unsqueeze(2)), dim=2)
    xy_couples = xy_couples.reshape(-1, 2)
    return xy_couples


class spdc_dataset(Dataset):

    def __init__(self, pump_crystal_file, coincidence_rate_file, idler_signa_coeffs_file):
        # input form : [z,{x,y} in sample space, basis_coefs, Avac(z=0)]
        self.N = hyperparams['N']
        self.sample_space = hyperparams['sample_space']
        self.xy_couples = get_xy_couples(160e-6, 160e-6, 4e-6, 4e-6).to(torch.float32)
        # self.pump_crystal = torch.from_numpy(pump_crystal_file).reshape(
        #     [pump_crystal_file.shape[0], hyperparams['basis_coefficients_pump'] +
        #      hyperparams['basis_coefficients_crystal']]).to(
        #     torch.float32)
        # without crystal coeffs
        self.pump_crystal = torch.from_numpy(pump_crystal_file[:, 0]).reshape(
            [pump_crystal_file.shape[0], hyperparams['basis_coefficients_pump']]).to(
            torch.float32)
        self.coincidence_rate = torch.from_numpy(coincidence_rate_file).to(torch.float32)
        self.idler_signal_coeffs = torch.from_numpy(idler_signa_coeffs_file).reshape(idler_signa_coeffs_file.shape[0], 2
                                                                                     , self.sample_space)
        self.coincidence_rate_mean = self.coincidence_rate.mean()
        self.coincidence_rate_std = self.coincidence_rate.std()

    def __len__(self):
        return self.coincidence_rate.shape[0]

    def __getitem__(self, idx):
        # shape: [z, (x, y), coeffs, normal_variable] - 144
        out = torch.ones([self.N, self.sample_space, 1]) * hyperparams['z_0']
        out = torch.cat((out, self.xy_couples.repeat(hyperparams['N'], 1, 1)), dim=2)
        out = torch.cat((out, self.pump_crystal[idx].repeat(hyperparams['N'], hyperparams['sample_space'], 1)), dim=2)
        normal_dist = torch.randn(hyperparams['N'], hyperparams['sample_space'], 1)
        out = torch.cat((out, normal_dist), dim=2)
        return ((out-out.mean())/out.std()), self.coincidence_rate[idx], self.idler_signal_coeffs[idx]


def grads_for_function(y, x):
    key_counter = 0
    output_dict = {'Ai_out': [], 'As_out': [], 'Ai_vac': [], 'As_vac': []}
    for key in output_dict:  # 3 derivatives for each sample.
        for _ in range(2):  # real/imaginary

            mask = torch.zeros_like(y)
            mask[:, :, :, key_counter] = 1
            dA_dx = grad(y, x, grad_outputs=mask, retain_graph=True, create_graph=True)[0]

            mask = torch.zeros_like(dA_dx)
            mask[:, :, :, 1] = 1
            d2A_dx2 = grad(dA_dx, x, grad_outputs=mask, create_graph=True)[0]
            d2A_dx2 = d2A_dx2[:, :, :, 1].unsqueeze(3)

            mask = torch.zeros_like(dA_dx)
            mask[:, :, :, 2] = 1
            d2A_dy2 = grad(dA_dx, x, grad_outputs=mask, create_graph=True)[0]
            d2A_dy2 = d2A_dy2[:, :, :, 2].unsqueeze(3)

            dA_dz = dA_dx[:, :, :, 0].unsqueeze(3)

            output_dict[key].append({'dz': dA_dz, 'd2x': d2A_dx2, 'd2y': d2A_dy2})
            key_counter += 1

    return output_dict


def func_calc(y, x, grad_dict, kappa_i, kappa_s):
    # kappas shape repeat.
    kappa_i = kappa_i.repeat(1, x.shape[1], 1, 1)
    kappa_s = kappa_s.repeat(1, x.shape[1], 1, 1)
    Ai_out_grads = grad_dict['Ai_out']
    As_vac_grads = grad_dict['As_vac']
    Ai_vac_grads = grad_dict['Ai_vac']
    As_out_grads = grad_dict['As_out']
    Ai_out = (y[:, :, :, 0] + 1j * y[:, :, :, 1]).unsqueeze(3)
    As_out = (y[:, :, :, 2] + 1j * y[:, :, :, 3]).unsqueeze(3)
    Ai_vac = (y[:, :, :, 4] + 1j * y[:, :, :, 5]).unsqueeze(3)
    As_vac = (y[:, :, :, 6] + 1j * y[:, :, :, 7]).unsqueeze(3)

    delta_k = hyperparams['kp'] - hyperparams['ks'] - hyperparams['ki']

    f1 = (1j * (Ai_out_grads[0]['dz'] + 1j * Ai_out_grads[1]['dz']) +
          ((Ai_out_grads[0]['d2x'] + 1j * Ai_out_grads[1]['d2x'] + Ai_out_grads[0]['d2y'] + 1j * Ai_out_grads[1]['d2y'])
           / (2 * hyperparams['ki']))
          - kappa_i * torch.exp(-1j * delta_k * x[:, :, :, 0].unsqueeze(3)) * torch.conj(As_vac))

    f2 = (1j * (As_out_grads[0]['dz'] + 1j * As_out_grads[1]['dz']) +
          (As_out_grads[0]['d2x'] + 1j * As_out_grads[1]['d2x'] + As_out_grads[0]['d2y'] + 1j * As_out_grads[1][
              'd2y'])
          / (2 * hyperparams['ks'])
          - kappa_s * torch.exp(-1j * delta_k * x[:, :, :, 0].unsqueeze(3)) * torch.conj(Ai_vac))

    f3 = (1j * (Ai_vac_grads[0]['dz'] + 1j * Ai_vac_grads[1]['dz']) +
          (Ai_vac_grads[0]['d2x'] + 1j * Ai_vac_grads[1]['d2x'] + Ai_vac_grads[0]['d2y'] + 1j * Ai_vac_grads[1][
              'd2y'])
          / (2 * hyperparams['ki'])
          - kappa_i * torch.exp(-1j * delta_k * x[:, :, :, 0].unsqueeze(3)) * torch.conj(As_out))

    f4 = (1j * (As_vac_grads[0]['dz'] + 1j * As_vac_grads[1]['dz']) +
          (As_vac_grads[0]['d2x'] + 1j * As_vac_grads[1]['d2x'] + As_vac_grads[0]['d2y'] + 1j * As_vac_grads[1][
              'd2y'])
          / (2 * hyperparams['ks'])
          - kappa_s * torch.exp(-1j * delta_k * x[:, :, :, 0].unsqueeze(3)) * torch.conj(Ai_out))

    return f1.squeeze(), f2.squeeze(), f3.squeeze(), f4.squeeze()


class forward_model(nn.Module):
    def __init__(self):
        super().__init__()
        # Physics_informed_MLP
        # input form : [z,{x,y} in sample space, basis_coeffs, Avac(z=0)-random gaussian]
        # output form : [Ai(out), As(out), Ai(vac), As(vac)] - Real and imaginary

        self.input_size_MLP = 1 + 2 + hyperparams['basis_coefficients_pump'] + hyperparams[
            'basis_coefficients_crystal'] + 1

        self.output_size_MLP = 4 * 2

        self.Physics_informed_MLP = nn.Sequential(
            nn.Linear(self.input_size_MLP, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            # nn.Dropout(0.2),
            nn.Linear(128, self.output_size_MLP)
        )

        # self.Physics_informed_MLP = nn.Sequential(
        #     nn.Linear(self.input_size_MLP, 32),
        #     nn.Tanh(),
        #     nn.Linear(32, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 128),
        #     nn.Tanh(),
        #     nn.Linear(128, 256),
        #     nn.Tanh(),
        #     nn.Linear(256, 512),
        #     nn.Tanh(),
        #     nn.Linear(512, 1024),
        #     nn.Tanh(),
        #     nn.Linear(1024, 512),
        #     nn.Tanh(),
        #     nn.Linear(512, self.output_size_MLP)
        # )

        # Correlations calculator model
        # input form : [Ai(out), As(out), Ai(vac)] + real/imaginary dimension
        # output form : [G_(2)] - Coincidence rate

        self.input_size_Calc = 3 * 2 * hyperparams['sample_space'] * hyperparams['N']
        self.output_size_Calc = hyperparams['projection_basis_size']

        self.Correlations_calculator = nn.Sequential(
            nn.Linear(self.input_size_Calc, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_size_Calc)
        )

    def forward(self, x):
        # A_is = self.Physics_informed_MLP(x)
        # corr = self.Correlations_calculator(
        #     A_is[:, :, :, :-2].reshape([-1, hyperparams['N'] * hyperparams['sample_space'] * 6]))
        # corr = corr/corr.abs().sum()
        A_is = self.Physics_informed_MLP(x).reshape(x.shape[0], hyperparams['N'], 81, 81, -1)
        Ai_out = (A_is[:, :, :, :, 0] + 1j * A_is[:, :, :, :, 1])
        As_out = (A_is[:, :, :, :, 2] + 1j * A_is[:, :, :, :, 3])
        Ai_vac = (A_is[:, :, :, :, 4] + 1j * A_is[:, :, :, :, 5])
        # corr = []
        # for i in range(x.shape[0]):
        #     obs = calc_observables(Ai_out[i], As_out[i], Ai_vac[i]).squeeze()
        #     obs = obs/obs.abs().sum()
        #     corr.append(obs)
        # corr = torch.stack(corr)
        obs = calc_observables(Ai_out[0], As_out[0], Ai_vac[0])
        obs = obs/obs.abs().sum()
        corr = obs
        return A_is, corr


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class Learner(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = forward_model()
        self.model.apply(init_weights)
        self.corr_loss = nn.L1Loss()
        self.lr = 1e-3

    def train_dataloader(self):
        pc = np.load('pump_crystal_coeffs_new_all_zero_except_pump.npy')[:10]
        cr = np.load('coincidence_rate_new_all_zero_except_pump.npy')[:10]
        isc = np.load('idler_signal_coeffs_new_all_zero_except_pump.npy')[:10]
        dataset = spdc_dataset(pc, cr, isc)
        trainloader = data.DataLoader(dataset, batch_size=1, num_workers=12)
        return trainloader

    def val_dataloader(self):
        pc = np.load('pump_crystal_coeffs_new_all_zero_except_pump.npy')[:10]
        cr = np.load('coincidence_rate_new_all_zero_except_pump.npy')[:10]
        isc = np.load('idler_signal_coeffs_new_all_zero_except_pump.npy')[:10]
        dataset = spdc_dataset(pc, cr, isc)
        val_loader = data.DataLoader(dataset, batch_size=1, num_workers=12)
        return val_loader

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def complex_loss(inputs):
        return F.mse_loss(inputs.abs(), torch.zeros_like(inputs.abs()))

    def residual_loss(self, A_i, inputs, kappas):
        functions = func_calc(A_i, inputs, grads_for_function(A_i, inputs), kappa_i=kappas[:, :1, :].unsqueeze(3),
                              kappa_s=kappas[:, 1:, :].unsqueeze(3))
        return Learner.complex_loss(functions[0]) + self.complex_loss(functions[1]) + self.complex_loss(
            functions[2]) + self.complex_loss(functions[3])

    def correlations_loss(self, corr, corr_target):
        return self.corr_loss(corr, corr_target)

    def training_step(self, batch, batch_idx):
        inputs, corr_targets, kappas = batch
        # inputs.requires_grad_()
        A_i, corr = self.model(inputs)
        # for GPU RAM savings we will compute the residual loss just for few of the samples.
        short_inputs = inputs[:, :1, :, :]
        # indices = torch.randint(short_inputs.shape[2], (6400,))
        # short_inputs = short_inputs[:, :, indices, :]
        short_inputs.requires_grad_()
        A_i_short = self.model.Physics_informed_MLP(short_inputs)
        residual_loss = self.residual_loss(A_i_short, short_inputs, kappas)
        correlations_loss = self.correlations_loss(corr.to(torch.float32), corr_targets.to(torch.float32))

        # TensorBoard logs.
        self.log('Train_Loss', residual_loss + correlations_loss)
        self.log('Correlations_Loss', correlations_loss)
        self.log('Residual_Loss', residual_loss)

        # weighting the loss based on observation of the loss graph
        return residual_loss + correlations_loss
        # return correlations_loss

    def validation_step(self, batch, batch_idx):
        inputs, corr_targets, kappas = batch
        _, corr = self.model(inputs)
        return self.correlations_loss(corr, corr_targets)

    def validation_epoch_end(self, validation_step_outputs):
        mean_loss = torch.stack(validation_step_outputs).mean()
        print(f'\nValidation loss: {mean_loss.item()}')
        self.log("validation_correlation_loss", mean_loss.item())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        # return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer),
                "monitor": "validation_correlation_loss",
                "frequency": 1
            },
        }


def main(new=True):
    if new:
        model = Learner()
    else:
        model = Learner.load_from_checkpoint('/home/barak/Documents/Project/project_on_pycharm/lightning_logs'
                                             '/version_0/checkpoints/epoch=164-step=1649.ckpt')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='validation_correlation_loss')
    trainer = pl.Trainer(gpus=1, accumulate_grad_batches=1, log_every_n_steps=10, callbacks=[checkpoint_callback],
                         auto_lr_find=True)
    # trainer.tune(model)
    trainer.fit(model)


if __name__ == "__main__":
    main()
