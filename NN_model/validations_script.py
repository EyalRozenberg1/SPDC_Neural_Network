from enangelment import Learner, spdc_dataset
import torch.utils.data as data
import pytorch_lightning as pl
import numpy as np




model = Learner.load_from_checkpoint('/home/barak/Documents/Project/project_on_pycharm/lightning_logs'
                                     '/version_0/checkpoints/epoch=2-step=163.ckpt')

pc = np.load('pump_crystal_coeffs_new_7.npy')
cr = np.load('coincidence_rate_new_7.npy')
isc = np.load('idler_signal_coeffs_new_7.npy')
dataset = spdc_dataset(pc, cr, isc)
trainloader = data.DataLoader(dataset, batch_size=1, num_workers=24)


trainer = pl.Trainer(gpus=1, accumulate_grad_batches=1, log_every_n_steps=1)

trainer.validate(model, trainloader)