import os
from glob import glob
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker


def get_energy_rmse(f_in_name, f_out_name, num_atoms):
    f_in = open(f_in_name, 'r')
    f_out = open(f_out_name, 'r')

    energy_read_flag = False
    energies_in = []
    energies_out = []

    for line in f_in:
        nline = line.split()
        if 'Energy' in nline:
            energy_read_flag = True
            continue
        if energy_read_flag:
            energies_in.append(float(nline[0])/num_atoms)
            energy_read_flag = False

    for line in f_out:
        nline = line.split()
        if 'Energy' in nline:
            energy_read_flag = True
            continue
        if energy_read_flag:
            energies_out.append(float(nline[0])/num_atoms)
            energy_read_flag = False

    energies_in = np.asarray(energies_in)
    energies_out = np.asarray(energies_out)

    # Calculate multiple error metrics
    mse = np.square(np.subtract(energies_in, energies_out)).mean()
    rmse = math.sqrt(mse)
    mae = np.abs(np.subtract(energies_in, energies_out)).mean()

    # Calculate R² score
    ss_res = np.sum((energies_in - energies_out) ** 2)
    ss_tot = np.sum((energies_in - np.mean(energies_in)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    f_in.close()
    f_out.close()

    print(f"Root Mean Square Error: {rmse:.6f} eV/atom")
    print(f"Mean Absolute Error: {mae:.6f} eV/atom")
    print(f"R² Score: {r2:.6f}")

    return energies_in, energies_out, rmse, mae, r2


directories = glob("./*/")

for directory in directories:
    print(f"Processing directory: {directory}")

    # Get error metrics for both test and train
    energies_test_dft, energies_test_mlip, rmse_test, mae_test, r2_test = get_energy_rmse(
        f_in_name='test.cfg', f_out_name=f"{os.path.join(directory, 'test_out.cfg')}", num_atoms=56
    )

    energies_train_dft, energies_train_mlip, rmse_train, mae_train, r2_train = get_energy_rmse(
        f_in_name='train.cfg', f_out_name=f"{os.path.join(directory, 'train_out.cfg')}", num_atoms=56
    )

    params = {
        'figure.figsize': (5, 5),
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.linewidth': 0.5,
        'grid.color': 'grey',
        'grid.alpha': 0.1,
        'grid.linewidth': 0.5,
        'figure.dpi': 250
    }

    matplotlib.pyplot.rcParams.update(params)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    # Scatter plots
    plt.scatter(energies_test_mlip, energies_test_dft, s=9, c='tab:orange', marker='o', zorder=1, alpha=0.5, label='Test')
    plt.scatter(energies_train_mlip, energies_train_dft, s=9, c='tab:blue', marker='d', zorder=1, alpha=0.5, label='Train')

    # Perfect prediction line
    plt.plot(np.arange(-10, 10), np.arange(-10, 10), linestyle='--', color='black', zorder=2)

    ax.tick_params(which='minor', direction='in')
    ax.tick_params(direction='in', length=6, width=0.5, colors='black')
    ax.set(xlabel='MLIP energies (eV/atom)', ylabel=r'DFT energies (eV/atom)')

    # Plot parameters
    ax.set_xlim(min(energies_test_dft) - 0.01, max(energies_test_dft) + 0.01)
    ax.set_ylim(min(energies_test_dft) - 0.01, max(energies_test_dft) + 0.01)

    # Display error metrics on the plot
    text_str = '\n'.join([
        f'Test Set:',
        f'RMSE: {rmse_test*1000:.2f} meV/atom',
        f'MAE: {mae_test*1000:.2f} meV/atom',
        f'R²: {r2_test:.4f}',
        f'',
        f'Train Set:',
        f'RMSE: {rmse_train*1000:.2f} meV/atom',
        f'MAE: {mae_train*1000:.2f} meV/atom',
        f'R²: {r2_train:.4f}'
    ])

    # Position the text box in upper left
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    # Alternative: Position in lower right if upper left is crowded
    # ax.text(0.95, 0.05, text_str, transform=ax.transAxes, fontsize=10,
    #         verticalalignment='bottom', horizontalalignment='right', bbox=props)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))

    ax.legend( fontsize=14)

    plt.grid()
    plt.savefig(f"{os.path.join(directory, 'energy_test.png')}", bbox_inches='tight')
    # plt.show()

    print(f"Plot saved to: {os.path.join(directory, 'energy_test.png')}\n")
