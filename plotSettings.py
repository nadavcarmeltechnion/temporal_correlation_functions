import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
# plt.rcParams.update({
#     # Font and text settings
#     'font.size': 5,  # General font size
#     'axes.labelsize': 6,  # Font size for axis labels
#     'axes.titlesize': 6,  # Font size for the plot title
#     'legend.fontsize': 5,  # Font size for legend
#     'xtick.labelsize': 5,  # Font size for x-axis tick labels
#     'ytick.labelsize': 5,  # Font size for y-axis tick labels
#     'font.family': 'serif',  # Use serif fonts
#     'font.serif': ['Times New Roman'],  # Specify a specific serif font (e.g., Times New Roman)
#
#     # Line and marker settings
#     'lines.linewidth': 0.7,  # Increase the line width for better visibility
#     'lines.markersize': 4,  # Default marker size
#     'lines.markeredgewidth': 0.05,  # Edge width of markers
#
#     # Figure and subplot settings
#     'figure.figsize': [3,2],  # Default figure size (width, height) in inches
#     'figure.dpi': 300,  # Dots per inch for the figure (better quality for papers)
#     'savefig.dpi': 300,  # DPI for saved figures (high quality for publication)
#     'figure.autolayout': True,  # Automatically adjust layout to avoid clipping of labels
#
#     # Grid settings
#     'axes.grid': True,  # Turn on the grid by default
#     'grid.linestyle': '--',  # Dotted grid lines
#     'grid.linewidth': 0.25,  # Grid line width
#     'grid.alpha': 0.35,  # Transparency of grid lines
#
#     # Axes and tick settings
#     'axes.linewidth': 0.5,  # Width of the axes lines
#     'xtick.major.size': 5,  # Length of major x-axis ticks
#     'ytick.major.size': 5,  # Length of major y-axis ticks
#     'xtick.minor.size': 3,  # Length of minor x-axis ticks
#     'ytick.minor.size': 3,  # Length of minor y-axis ticks
#     'xtick.direction': 'in',  # Ticks inside the plot
#     'ytick.direction': 'in',
#
#     # Legend settings
#     'legend.frameon': True,  # Add a frame around the legend
#     'legend.loc': 'best',  # Automatically choose the best location for the legend
#     'legend.framealpha': 0.9,  # Legend frame transparency
#     'legend.fancybox': True,  # Use rounded corners in the legend frame
#     'legend.borderpad': 0.5,  # Padding around legend content
# })

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
print(matplotlib.get_backend())
# matplotlib.use('TkAgg')



def plot_2d(data, x=None, y=None, logscale=False, cmap='viridis', vmin=None, vmax=None, colorbar_label='', title='',xlabel='',ylabel=''):
    """
    Plot a 2D array using pcolormesh.

    Parameters:
        data (2D array): The values to plot.
        x, y (1D arrays): Coordinates for the x and y axes. If None, uses indices.
        logscale (bool): Whether to apply log scale on the color axis.
        cmap (str): Matplotlib colormap.
        vmin, vmax: Color scale limits.
        colorbar_label (str): Label for the colorbar.
        title (str): Title of the plot.
    """
    data = np.array(data)

    if x is None:
        x = np.arange(data.shape[1] + 1)
    if y is None:
        y = np.arange(data.shape[0] + 1)

    X, Y = np.meshgrid(x, y)

    norm = LogNorm(vmin=vmin, vmax=vmax) if logscale else None

    plt.figure()
    pcm = plt.pcolormesh(X, Y, data, cmap=cmap, norm=norm, shading='auto')
    cbar = plt.colorbar(pcm)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    # plt.axes('equal')
    plt.show()

def plot_1d(x, y, fx = None, logx=False, logy=False, xlabel='', ylabel='', title='', label=None, show=False, yerr=None, ax=None,color=None,marker=None):
    """
    Plot 1D data with optional log axes.

    Parameters:
        x, y: 1D arrays of the same length.
        logx, logy: Apply log scale to x/y axis.
        xlabel, ylabel, title: Strings for labels and title.
        label: Optional legend label.
    """
    # plt.figure()
    if ax is None:
        if fx is None:
            if yerr is None:
                plt.scatter(x, y, label=label,color=color,marker=marker)
            else:
                plt.errorbar(x, y, yerr=yerr, fmt='o', label=label,color=color)
        else:
            if yerr is None:
                plt.scatter(fx(x), y, label=label,color=color,marker=marker)
            else:
                plt.errorbar(fx(x), y, yerr=yerr, fmt='o', label=label,color=color)
    else:
        if fx is None:
            if yerr is None:
                plt.scatter(x, y, label=label,color=color,marker=marker)
            else:
                plt.errorbar(x, y, yerr=yerr, fmt='o', label=label,color=color)
        else:
            if yerr is None:
                plt.scatter(fx(x), y, label=label,color=color,marker=marker)
            else:
                plt.errorbar(fx(x), y, yerr=yerr, fmt='o', label=label,color=color)
            # Set tick positions and labels manually
            tick_x = [x[0],x[len(x)//100-1],x[len(x)//10-1],x[-1]]  # ticks in x (original space)
            tick_fx = fx(tick_x)  # corresponding f(x) positions on the axis
            ax.set_xticks(tick_fx)
            ax.set_xticklabels([f"{val:.1f}" for val in tick_x])  # label as x, not f(x)


    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')
    if label:
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if show:
        plt.show()

def dual_axis_scatter_sets(
        x,
        sets,
        xlabel="X",
        ylabel1="Linear Y",
        ylabel2="Logarithmic Y"
):
    """
    Plot multiple data sets with shared x-axis.
    Each set contains (y_linear, y_log, label, color).

    Args:
        x (array-like): Shared x-axis values.
        sets (list of tuples): Each tuple is (y_linear, y_log, label, color).
        xlabel (str): Label for the x-axis.
        ylabel1 (str): Label for the linear y-axis (left).
        ylabel2 (str): Label for the log y-axis (right).
    """
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()

    for y_lin, y_log, y_lin_err, y_log_err, label, color in sets:
        ax1.errorbar(x, y_lin, yerr=y_lin_err, fmt='o', label=f"{label} (linear)", color=color)
        ax2.errorbar(x, y_log, yerr=y_log_err, fmt='*', label=f"{label} (log)", color=color)


    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color='black')
    ax2.set_ylabel(ylabel2, color='black')
    ax2.set_yscale('log')

    # Combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='best')

    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def plot_operator_sequence_with_slider(Os, cmap='binary', log=False):
    """
    Display a list/array of matrices as heatmaps with an interactive slider.

    Parameters:
        Os: list or np.array of shape (N, d, d), each [i] is a matrix to display
        cmap: colormap used for display (default 'seismic')
    """
    Os = np.array(Os)
    if log:
        Os = np.log(Os)
    N = Os.shape[0]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    # Initial image (0th operator)
    im = ax.imshow(Os[0].real, cmap=cmap, interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax)
    ax.set_title("Operator 0 (real part)")

    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Index', 0, N - 1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        im.set_data(Os[idx].real)
        ax.set_title(f"Operator {idx} (real part)")
        im.set_clim(vmin=np.min(Os.real), vmax=np.max(Os.real))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

