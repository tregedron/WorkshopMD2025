import os

import shutil
from glob import glob

import numpy as np

T_list = [800, 1000, 1200]
replicas = 3

initial_path = os.path.dirname(os.path.realpath(__file__))

for T in T_list:
    for replica in range(replicas):

        path = os.path.join(f"T_{T}", f"replica_{replica}")

        os.makedirs(path, exist_ok=True)
        shutil.copy("mlip.ini", path)
        shutil.copy("curr_16.mtp", path)
        shutil.copy("cubic56.data", os.path.join(path,"ini_conf.data"))

        with open ("./lammps.txt", "r") as fin:
            lammps_txt = fin.read()
            lammps_txt = lammps_txt.replace("! velocity all create 1400 123456 dist gaussian",
                                            f"velocity all create 1400 {np.random.randint(0, 123456)} dist gaussian")

            lammps_txt = lammps_txt.replace("! variable T equal 1000",
                                            f"variable T equal {T}")
        with open (os.path.join(path,"lammps.txt"), "w") as fout:
            fout.write(lammps_txt)

