import torch
from enangelment import Learner, spdc_dataset
import numpy as np
from results_and_stats_utils import save_results
import matplotlib.pyplot as plt


def main():
    inference_model = Learner.load_from_checkpoint('/home/barak/Documents/Project/project_on_pycharm/lightning_logs'
                                                   '/version_0/checkpoints/epoch=164-step=1649.ckpt')
    inference_model.cuda()
    inference_model.eval()
    while True:
        with torch.no_grad():
            # pc = np.load('/home/barak/Documents/Project/Git_SIM/SPDC_Neural_Network/results/test0/pump_crystal_coeffs.npy')[
            #     None, ...]
            # isc = \
            # np.load('/home/barak/Documents/Project/Git_SIM/SPDC_Neural_Network/results/test0/idler_signal_coeffs.npy')[
            #     None, ...]
            # cr = np.load('/home/barak/Documents/Project/Git_SIM/SPDC_Neural_Network/results/test0/coincidence_rate.npy')[
            #     None, ...]
            pc = np.load('pump_crystal_coeffs_new_all_zero_except_pump.npy')
            cr = np.load('coincidence_rate_new_all_zero_except_pump.npy')
            isc = np.load('idler_signal_coeffs_new_all_zero_except_pump.npy')
            dataset = spdc_dataset(pc, cr, isc)
            sample = dataset[3]
            As_out, output = inference_model(sample[0].unsqueeze(0).cuda())
            As_out = As_out[0, 0, :, 0] + 1j * As_out[0, 0, :, 1]
            As_out = As_out.abs()
            save_results('test_best_model', output)
            plt.imshow((output.cpu().reshape(9, 9).numpy()))
            plt.show()


if __name__ == "__main__":
    main()
