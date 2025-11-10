import os

from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("summary.csv")

result_mean = df.groupby(['T']).mean(numeric_only=True).reset_index()
result_std = df.groupby(['T']).std(numeric_only=True).reset_index()

plt.plot([599.146514935989,799.715504978663,1000.28449502134,1203.41394025605], [2.20408163265306,2.07959183673469,1.95510204081633,1.82789115646259], marker="x", linestyle = "-", label=f"Exp Romatoski")
plt.errorbar(result_mean["T"], result_mean["Dens, g/cm^3"], result_std["Dens, g/cm^3"], marker="o", linestyle = "-", ecolor="black", capsize=3, label=f"MTP-MD")
plt.title(f"Density vs T, FLiNaK", fontsize=16)
plt.xlabel(f"Temperature, K", fontsize=16)
plt.ylabel(r"Density, g/cm$^3$", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig(f"density.png", dpi=300)
plt.close()


plt.plot(result_mean["T"], result_mean["D_F, cm^2/s"], marker="o", linestyle = "-", label=f"F")
plt.plot(result_mean["T"], result_mean["D_Li, cm^2/s"], marker="o", linestyle = "-", label=f"Li")
plt.plot(result_mean["T"], result_mean["D_Na, cm^2/s"], marker="o", linestyle = "-", label=f"Na")
plt.plot(result_mean["T"], result_mean["D_K, cm^2/s"], marker="o", linestyle = "-", label=f"K")
plt.title(f"Diffusion vs T, FLiNaK", fontsize=16)
plt.xlabel(f"Temperature, K", fontsize=16)
plt.ylabel(r"Diffusion, cm$^2$/s", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig(f"diffusion.png", dpi=300)
plt.close()
