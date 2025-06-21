import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_regular_barplot(x: Union[list, np.ndarray], y: Union[list, np.ndarray], save_path: Union[str, None] = None,
                         **kwargs):
    """
    Creates a regular bar plot using the provided x and y data.

    Parameters
    ----------
    x : list or np.ndarray
        The categories or x-axis values for the bar plot.
    y : list or np.ndarray
        The heights or y-axis values for the bars.
    save_path : str or None, optional
        File path to save the plot image. If None (default), the plot is not saved.
    **kwargs
        Additional keyword arguments passed to the plotting function, such as:
            - figsize: tuple, figure size
            - color: str, bar color
            - xlabel, ylabel, title: str, axis and plot labels
            - label_fontdict: dict, font properties for axis labels
            - title_fontsize: int, font size for the title
            - xticks, yticks: list, custom tick locations
            - xticks_fontsize, yticks_fontsize: int, font size for ticks
            - xticks_rotation, yticks_rotation: int or float, rotation for ticks
            - legend: bool, whether to show legend
            - legend_loc: str, legend location
            - legend_fontsize: int, legend font size
            - is_tight_layout: bool, whether to use tight layout
    """
    plt.figure(figsize=kwargs.get('figsize', (10, 10)))
    plt.bar(x, y, color=kwargs.get('color', 'blue'))
    plt.xlabel(kwargs['xlabel'] if 'xlabel' in kwargs else None,
               fontdict=kwargs.get('label_fontdict', None))
    plt.ylabel(kwargs['ylabel'] if 'ylabel' in kwargs else None,
               fontdict=kwargs.get('label_fontdict', None))
    plt.title(kwargs['title'] if 'title' in kwargs else None,
              fontsize=kwargs['title_fontsize'] if 'fontsize' in kwargs else None)
    plt.xticks(kwargs['xticks'] if 'xticks' in kwargs else None,
               fontsize=kwargs['xticks_fontsize'] if 'xticks_fontsize' in kwargs else None,
               rotation=kwargs['xticks_rotation'] if 'xticks_rotation' in kwargs else None)
    plt.yticks(kwargs['yticks'] if 'yticks' in kwargs else None,
               fontsize=kwargs['yticks_fontsize'] if 'yticks_fontsize' in kwargs else None,
               rotation=kwargs['yticks_rotation'] if 'yticks_rotation' in kwargs else None)
    if kwargs.get('legend', False):
        plt.legend(loc=kwargs['legend_loc'] if 'legend_loc' in kwargs else None,
                   fontsize=kwargs['legend_fontsize'] if 'legend_fontsize' in kwargs else None)
    if kwargs.get('is_tight_layout', False):
        plt.tight_layout()
    if save_path is not None:
        if os.path.exists(save_path):
            raise FileExistsError(save_path)
        plt.savefig(save_path)
    plt.show()