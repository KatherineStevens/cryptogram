from typing import NoReturn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from errors import UnknownGraph
from main import ALPHA


def show_frequency(single_freq: dict, show: bool = True) -> NoReturn:
    """Show the graph of single frequency dictionary.

    Args:
        single_freq (dict): Single letter frequencies of text.
        show (bool): Displays graph.

    Returns:
        (NoReturn)
    """
    # split up the dictionary so that the x axis is list of keys (letters) and y (frequencies) axis is list of values
    x, y = [], []
    for key in sorted(single_freq.keys()):
        x.append(key)
        y.append(single_freq[key])

    fig, ax = plt.subplots(1, 1)  # create scatter plot and label axis
    ax.scatter(x, y)
    ax.set_xlabel("Letter")
    ax.set_ylabel("Frequency")

    if show:
        plt.show()  # show graph


def show_frequency_words(text: str, show: bool = True) -> NoReturn:
    """Show the graph of single frequency dictionary.

    Args:
        text (str): String text.
        show (bool): Displays graph.

    Returns:
        NoReturn
    """
    text = text.upper()
    word_count = Counter(
        text.split()
    )  # use collections counter to count words and create dictionary

    # split up the dictionary so that the x axis is list of keys (words) and y (frequencies) axis is list of values
    x = sorted(word_count, key=lambda z: word_count[z], reverse=True)
    y = sorted(word_count.values(), reverse=True)

    fig, ax = plt.subplots(1, 1)  # create scatter plot and label axis
    ax.scatter(x[:15], y[:15])
    ax.set_xlabel("Word")
    ax.set_ylabel("Frequency")

    if show:
        plt.show()  # show graph


def show_heat_map(bigram: np.ndarray, show: bool = True) -> NoReturn:
    """Show the heat map of a bigram matrix.

    Args:
        bigram (np.ndarray): Bigram frequencies of text.
        show (bool): Displays graph.

    Returns:
        NoReturn
    """
    ax = sns.heatmap(
        bigram, xticklabels=ALPHA, yticklabels=ALPHA
    )  # use seaborn heat map using alphabet for axis
    plt.yticks(rotation=0)  # rotates y tick labels
    ax.invert_yaxis()  # makes alphabet increase from bottom

    if show:
        plt.show()  # show graph


def save_graph(
    file: str, single_freq: dict = None, bigram: np.ndarray = None
) -> NoReturn:
    """Save a frequency or heat map graph.

    Args:
        file (str): Desired file name.
        single_freq (dict): Single letter frequencies of text.
        bigram (np.ndarray): Bigram frequencies of text.

    Returns:
        NoReturn
    """
    if single_freq:  # specify which type, create the plot but set show to False
        show_frequency(single_freq, False)
    elif np.any(bigram):
        show_heat_map(bigram, False)
    else:
        raise UnknownGraph

    plt.savefig(f"img/{file}", dpi=300)  # save with resolution 300 pixels per inch
