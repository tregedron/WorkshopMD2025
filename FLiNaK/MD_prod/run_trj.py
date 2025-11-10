import os

from glob import glob

path2lmp = "lammps"
cores = 2

run_dirs = glob("./T_*/replica_*")
initial_path = os.path.dirname(os.path.realpath(__file__))

for run_dir in run_dirs:

    print(f"Starting MD modeling in {run_dir}.")

    os.chdir(f'{run_dir}')
    os.system(f"mpirun -n {cores} {path2lmp} -in lammps.txt")
    os.chdir(f'{initial_path}')

    print(f"Finished MD modeling in {run_dir}!")
