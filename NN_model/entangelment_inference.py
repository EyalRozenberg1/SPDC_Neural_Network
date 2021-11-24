import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from enangelment import Learner, get_xy_couples, hyperparams, spdc_dataset


def create_example(coeffs):
    out = torch.ones(hyperparams['sample_space'], 1) * hyperparams['z_0']
    out = torch.cat((out, get_xy_couples(160e-6, 160e-6, 4e-6, 4e-6).to(torch.float32)), dim=1)
    out = torch.cat((out, coeffs.repeat(hyperparams['sample_space'], 1)), dim=1)

    normal_dist = torch.randn(hyperparams['sample_space'], 1)
    out = torch.cat((out, normal_dist), dim=1)
    return out


def main():
    inference_model = Learner.load_from_checkpoint('/home/barak/Documents/Project/project_on_pycharm/lightning_logs'
                                                     '/version_6/checkpoints/epoch=18-step=189.ckpt')
    inference_model.eval()
    while True:
        coeffs = torch.randn(28)
        input = create_example(coeffs=coeffs)
        input = ((input - input.mean())/input.std())
        output = inference_model.model.Physics_informed_MLP(input)
        As_out = output[:, 0] + 1j * output[:, 1]
        As_out = As_out.abs()
        plt.imshow((As_out.detach().reshape(81, 81).numpy()), cmap='Greys', interpolation='bilinear', vmin=0, vmax=1)
        plt.show()
        # interpolation = 'bilinear'


if __name__ == "__main__":
    main()
