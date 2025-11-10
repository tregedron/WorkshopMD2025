import os

from glob import glob

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker

def get_forces_rmse(f_in_name, f_out_name):
    f_in = open(f_in_name, 'r')
    f_out = open(f_out_name, 'r')

    forces_read_flag = False
    forces_in = []
    forces_out = []

    for line in f_in:
        nline = line.split()
        if 'AtomData:' in nline:
            forces_read_flag = True
            continue
        elif 'Energy' in nline:
            forces_read_flag = False

        if forces_read_flag:
            forces_in.append([float(nline[-3]), float(nline[-2]), float(nline[-1])])

    f_in.close()

    for line in f_out:
        nline = line.split()
        if 'AtomData:' in nline:
            forces_read_flag = True
            continue
        elif 'Energy' in nline:
            forces_read_flag = False

        if forces_read_flag:
            forces_out.append([float(nline[-3]), float(nline[-2]), float(nline[-1])])

    f_out.close()

    forces_in = np.asarray(forces_in)
    forces_out = np.asarray(forces_out)

    print(f"Forces shape: {forces_in.shape}")

    # Calculate multiple error metrics
    mse = np.square(np.subtract(forces_in, forces_out)).mean()
    rmse = math.sqrt(mse)
    mae = np.abs(np.subtract(forces_in, forces_out)).mean()

    # Calculate R² score
    ss_res = np.sum((forces_in - forces_out) ** 2)
    ss_tot = np.sum((forces_in - np.mean(forces_in)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    print(f"Root Mean Square Error: {rmse:.6f} eV/Å")
    print(f"Mean Absolute Error: {mae:.6f} eV/Å")
    print(f"R² Score: {r2:.6f}")

    return forces_in, forces_out, rmse, mae, r2

directories = glob("./*/")

for directory in directories:
    print(f"Processing directory: {directory}")

    # Get error metrics for both test and train
    forces_dft_test, forces_mlip_test, rmse_test, mae_test, r2_test = get_forces_rmse(f_in_name='test.cfg', f_out_name=f"{os.path.join(directory, 'test_out.cfg')}")
    forces_dft_train, forces_mlip_train, rmse_train, mae_train, r2_train = get_forces_rmse(f_in_name='train.cfg', f_out_name=f"{os.path.join(directory, 'train_out.cfg')}")

    # Flatten the forces arrays for plotting
    forces_dft_test_flat = forces_dft_test.flatten()
    forces_mlip_test_flat = forces_mlip_test.flatten()
    forces_dft_train_flat = forces_dft_train.flatten()
    forces_mlip_train_flat = forces_mlip_train.flatten()

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

    plt.scatter(forces_mlip_test_flat, forces_dft_test_flat, s=9, c='tab:orange', marker='o', zorder=1, alpha = 0.5, label='Test')
    plt.scatter(forces_mlip_train_flat, forces_dft_train_flat, s=9, c='tab:blue', marker='d', zorder=1, alpha = 0.5, label='Train')

    plt.plot(np.arange(-10, 10), np.arange(-10, 10), linestyle='--', color='black', zorder=2)

    ax.tick_params(which='minor', direction='in')
    ax.tick_params(direction='in', length=6, width=0.5, colors='black')
    ax.set(xlabel='MLIP forces (eV/Å)', ylabel=r'DFT forces (eV/Å)')

    # Plot parameters
    ax.set_xlim(-5.6, 5.6)
    ax.set_ylim(-5.6, 5.6)

    # Display error metrics on the plot
    text_str = '\n'.join([
        f'Test Set:',
        f'RMSE: {rmse_test*1000:.2f} meV/Å',
        f'MAE: {mae_test*1000:.2f} meV/Å',
        f'R²: {r2_test:.4f}',
        f'',
        f'Train Set:',
        f'RMSE: {rmse_train*1000:.2f} meV/Å',
        f'MAE: {mae_train*1000:.2f} meV/Å',
        f'R²: {r2_train:.4f}'
    ])

    # Position the text box in upper left
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    ax.legend(fontsize=14)

    plt.grid()
    plt.savefig(f"{os.path.join(directory, 'force_test.png')}", bbox_inches='tight')
    plt.close()
