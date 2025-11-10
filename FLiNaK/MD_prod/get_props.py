import os
import re

from glob import glob

import pandas as pd
import numpy as np

from scipy.optimize import curve_fit

def linear(x, a, b):
    return a*x + b

def parse_log(path):
    prop_dict = {}

    with open(path, "r") as logfile:

        names = []
        prop_dict["number"] = 0
        name2num = {}

        flag_prod = False

        for line_num, line in enumerate(logfile):

            if re.search(r"\d+\s+atoms\s+in\s+group\s+\w+", line):
                split = line.split()
                number = split[0]
                name =  split[-1]
                names.append(name)
                prop_dict["number"] += int(number)
                name2num[name] = int(number)

            if "variable T equal" in line:
                prop_dict["T"] = float(line.split()[-1])
            if "variable P equal" in line:
                prop_dict["P"] = float(line.split()[-1])

            if "fix production" in line:
                flag_prod = True

            if flag_prod and ("Step" in line):
                start_line = line_num
            if flag_prod and ("Loop time of" in line):
                stop_line = line_num

    df_logfile = pd.read_table(path, skiprows=start_line, nrows=stop_line - start_line-1, sep="\s+")
    # T_avg = <T_sys>
    prop_dict["T_avg"] = np.average(df_logfile["Temp"])

    # T_avg = <T_sys>
    prop_dict["P_avg"] = np.average(df_logfile["Press"])

    prop_dict["Dens, g/cm^3"] = np.average(df_logfile["Density"])
    prop_dict["dDens, g/cm^3"] = np.std(df_logfile["Density"])

    prop_dict["V, A^3"] = np.average(df_logfile["Volume"])
    prop_dict["L, A"] = (np.average(df_logfile["Volume"]))**(1/3)

    start_fit = 10
    time = df_logfile["Step"][start_fit:].values

    # time to s
    time=time*10**(-15)

    for name in names:
        MSD = df_logfile[f"c_msd{name}[4]"][start_fit:].values

        # MSD to cm^2
        MSD = MSD*10**(-16)

        popt, pcov = curve_fit(linear, time, MSD)

        prop_dict[f"D_{name}, cm^2/s"] = popt[0]/6

    print(f"calculated everything in {path}")

    return prop_dict

if __name__ == "__main__":
    paths = glob("./T_*/replica_*/log.lammps")
    print(paths)

    general_df = pd.DataFrame()

    for path_to_log in paths:
        df_sys_properties = pd.DataFrame.from_dict([parse_log(path_to_log)])
        df_sys_properties.to_csv(os.path.join(*path_to_log.split("/")[:-1],"properties.csv"))
        try:

            general_df = pd.concat([general_df, df_sys_properties])
        except:
            print(f"Something gone wrong {path_to_log}")
    general_df = general_df.sort_values(["T"])
    general_df.to_csv("summary.csv", index=False)
