import os

from glob import glob

mlp_path = "mlp"

mtp_paths = glob("./*/*.mtp")

for mtp_path in mtp_paths:
    # evaluate the energies, forces and stresses
    os.system(f"{mlp_path} calc-efs {mtp_path} train.cfg {os.path.join(*mtp_path.split('/')[:-1], 'train_out.cfg')}")
    os.system(f"{mlp_path} calc-efs {mtp_path} test.cfg {os.path.join(*mtp_path.split('/')[:-1], 'test_out.cfg')}")
