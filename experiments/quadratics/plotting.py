import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from experiments.helper.for_plotting import set_size
import matplotlib.gridspec as gridspec


def create_evaluation_plot(loading_path, path_of_experiment):

    width = 469.75499  # Arxiv
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts quantile_distance little smaller
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7
    }
    plt.rcParams.update(tex_fonts)

    names = {'std': 'HBF', 'pac': 'Learned', 'other': 'other'}
    colors = {'std': '#4cc9f0', 'pac': '#f72585', 'other': '#613f75', 'stopping_time': '#03dd5e',
              'conv_rate': '#ffda1f'}
    # colors = {'std': '#FF6F7F', 'pac': '#00D3FF', 'other': '#2A2B2DFF'}

    losses_std = np.load(loading_path + 'losses_of_baseline_algorithm.npy')
    losses_pac = np.load(loading_path + 'losses_of_learned_algorithm.npy')
    stopping_times_pac = np.load(loading_path + 'times_of_learned_algorithm.npy')
    rates = np.load(loading_path + 'rates_of_learned_algorithm.npy')

    n_train = np.load(loading_path + 'number_of_iterations.npy')
    pac_bound_rate = np.load(loading_path + 'pac_bound_rate.npy')
    upper_bound_rate = np.load(loading_path + 'upper_bound_rate.npy')
    pac_bound_times = np.load(loading_path + 'pac_bound_time.npy')
    upper_bound_times = np.load(loading_path + 'upper_bound_time.npy')

    # Create Figure
    q_l, q_u = 0.025, 0.975

    subplots = (6, 4)
    size = set_size(width=width, subplots=subplots)
    fig = plt.figure(figsize=size)
    G = gridspec.GridSpec(subplots[0], subplots[1])
    ax_0 = fig.add_subplot(G[0:3, :])
    ax_1 = fig.add_subplot(G[3:, 0:2])
    ax_2 = fig.add_subplot(G[3:, 2:])

    alpha = 0.5
    ax_1.hist(rates, alpha=alpha, bins=20, color=colors['conv_rate'], edgecolor=colors['conv_rate'])
    ax_1.axvline(np.mean(rates), color=colors['conv_rate'], linestyle='dashed')
    ax_1.axvline(np.median(rates), color=colors['conv_rate'], linestyle='dotted')
    ax_1.axvline(pac_bound_rate, 0, 1, color='#FF9B70', linestyle='dashed', label='PAC-bound')
    ax_1.axvline(upper_bound_rate, 0, 1, color='black', label='$r_{\\mathrm{max}}$', linestyle=(0, (1, 1)))
    ax_1.set(title=f'Conv. Rate', xlabel='$r(\\xi_n, \\theta_n)$')
    ax_1.grid('on')
    ax_1.legend()

    # Plot stopping times
    alpha = 0.5
    ax_2.hist(stopping_times_pac, alpha=alpha, bins=20, color=colors['stopping_time'], edgecolor=colors['stopping_time'])
    ax_2.axvline(np.mean(stopping_times_pac), color=colors['stopping_time'], linestyle='dashed')
    ax_2.axvline(np.median(stopping_times_pac), color=colors['stopping_time'], linestyle='dotted')
    ax_2.axvline(pac_bound_times, 0, 1, color='#02a144', linestyle='dashed', label='PAC-bound')
    ax_2.axvline(upper_bound_times, 0, 1, color='black', label='$t_{\\mathrm{max}}$', linestyle=(0, (1, 1)))
    ax_2.set(title=f'Conv. Time', xlabel='$\\tau_n$')
    ax_2.grid('on')
    ax_2.legend()

    # Compute mean and median for learned and standard losses
    mean_std, mean_pac = np.mean(losses_std, axis=0), np.mean(losses_pac, axis=0)
    median_std, median_pac = np.median(losses_std, axis=0), np.median(losses_pac, axis=0)
    iterations = np.arange(0, losses_std.shape[1])

    # Plot standard losses
    ax_0.plot(iterations, mean_std, color=colors['std'], linestyle='dashed', label=names['std'])
    ax_0.plot(iterations, median_std, color=colors['std'], linestyle='dotted')
    ax_0.fill_between(iterations, np.quantile(losses_std, q=q_l, axis=0), np.quantile(losses_std, q=q_u, axis=0),
                      color=colors['std'], alpha=0.5)

    # Plot pac losses
    ax_0.plot(iterations, mean_pac, color=colors['pac'], linestyle='dashed', label=names['pac'])
    ax_0.plot(iterations, median_pac, color=colors['pac'], linestyle='dotted')
    ax_0.fill_between(iterations, np.quantile(losses_pac, q=q_l, axis=0), np.quantile(losses_pac, q=q_u, axis=0),
                      color=colors['pac'], alpha=0.5)

    # Highlight the number of iterations the algorithm was trained for
    ax_0.axvline(n_train, 0, 1, color=colors['pac'], linestyle='dashdot', alpha=0.5, label='$n_{train}$')

    # Finalize plot for loss over iterations
    ax_0.set(title=f'Loss over Iterations', xlabel='$n_{it}$', ylabel='$\\ell(x^{(i)})$')
    ax_0.legend()
    ax_0.grid('on')
    ax_0.set_yscale('log')

    plt.tight_layout()
    fig.savefig(path_of_experiment + '/evaluation.pdf', dpi=300, bbox_inches='tight')
