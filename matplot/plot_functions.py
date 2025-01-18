import matplotlib.pyplot as plt
import numpy as np

def plot_vectors(vectors, colors, xlim=(0, 5), ylim=(0, 5)):
    """
    Plot one or more vectors in a 2D plane, specifying a color for each.

    Arguments
    ---------
    vectors: list of lists or arrays
        Coordinates of the vectors to plot. For example, [[1, 3], [2, 2]] 
        contains two vectors to plot, [1, 3] and [2, 2].
    colors: list
        Colors of the vectors. For instance: ['red', 'blue'] will display the
        first vector in red and the second in blue.
    xlim: tuple, optional
        x-axis limits (default is (0, 5)).
    ylim: tuple, optional
        y-axis limits (default is (-0, 5)).

    Example
    -------
    plot_vectors([[1, 3], [2, 2]], ['red', 'blue'])
    """
    plt.figure()
    plt.axvline(x=0, color='lightgray')  # X-axis line
    plt.axhline(y=0, color='lightgray')  # Y-axis line

    for i in range(len(vectors)):
        # Vectors are in the form of (x, y)
        x, y = vectors[i]
        plt.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1, color=colors[i])

    # Set x and y axis limits
    plt.xlim(xlim)
    plt.ylim(ylim)
    
    plt.grid(True)
    plt.show()
