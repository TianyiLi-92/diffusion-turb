import numpy as np
import h5py

filename = 'omega8_vxvyvz_forGan.h5'

with h5py.File(filename, 'r') as h5f:
    x = np.array(h5f['x'])

x = np.linalg.norm(x, axis=3, keepdims=True)
x = np.moveaxis(x, -1, 1)

x_train = x[:84480]
x_dev = x[84480:95040]

filename = 'omega8_vxvyvz_forGanValidation.h5'

with h5py.File(filename, 'r') as h5f:
    x = np.array(h5f['x'])

x = np.linalg.norm(x, axis=3, keepdims=True)
x = np.moveaxis(x, -1, 1)

x_test = x

rx0, rx1 = np.amin(x_train), np.amax(x_train)

x_train = 2*(x_train-rx0)/(rx1-rx0) - 1
x_dev   = 2*(x_dev-rx0)  /(rx1-rx0) - 1
x_test  = 2*(x_test-rx0) /(rx1-rx0) - 1

filename_out = 'TURB-Rot_new-data_module_diffusion.h5'

with h5py.File(filename_out, 'w') as hf:
    hf.create_dataset('min', data=rx0)
    hf.create_dataset('max', data=rx1)
    hf.create_dataset('train', data=x_train)
    hf.create_dataset('dev', data=x_dev)
    hf.create_dataset('test', data=x_test)
