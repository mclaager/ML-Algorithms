import numpy as np

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from matplotlib.pyplot import cm

from matplotlib.animation import FuncAnimation

def display_prediction_heatmap (model, x_min: float, x_max: float, y_min: float, y_max: float, acc: float) -> None:
    """Displays a heatmap of a binary classification model's predictions in a 2D grid.
    
    :param model: The model to generate 2D heatmap of. Must be able to make predictions
    based on an (x,y) input using a .predict() method
    :param x_min: Minimum x-value on grid
    :param x_max: Maximum x-value on grid
    :param y_min: Minimum y-value on grid
    :param y_max: Maximum y-value on grid
    :param acc: How accurate to make the heatmap, influences runtime
    """
    # Creates X, Y plane
    x = np.linspace(x_min, x_max, acc)
    y = np.linspace(y_min, y_max, acc)
    X, Y = np.meshgrid(x, y)
    # Creates Z values based on x, y grid
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z = np.column_stack((X_flat, Y_flat))
    Z = model.predict(Z)
    Z = Z.reshape((acc,acc))

    # Configures the plot
    fig = plt.figure()
    ax = plt.axes()

    # Draws the heatmap
    contours = ax.contourf(X, Y, Z, acc, cmap='RdBu_r')
    im = ax.imshow(Z, extent=[x_min, x_max, y_min, y_max],
                    origin = 'lower', cmap='RdBu_r', alpha=0.5, aspect='auto')

    # Gives a colorbar (should be in range [0,1])
    plt.colorbar(im, ax=ax)

    # Gives labels to the chart
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title("Heatmap Based on Network Predictions")
    plt.show()
    return

def display_data (data: np.ndarray, labels: np.ndarray, title = 'Plotted Data', \
    legend: bool = True) -> None:
    """
    Creates a plot for 2D multi-class classification data.
    
    :param data: The input dataset
    :param labels: The labels for the dataset
    :param legend: Whether to show a legend for the data (default: True)
    :param title: What the title for the plot will be
    """
    # Reshapes the labels to make for easier comparison
    idx = labels.flatten()
    # Seperate and plot the data according to label
    label_count = np.unique(idx).shape[0]
    split_data = np.array([data[idx == i,:] for i in range(label_count)])
    # Plots the data
    color = cm.rainbow(np.linspace(0, 1, label_count))
    for i,c in zip(range(label_count), color):
        plt.scatter(split_data[i][:,0], split_data[i][:,1], c=c.reshape(1,-1), label='Class {}'.format(i))
    # Adds some descriptive elements and shows the plot
    if legend:
        plt.legend()
    plt.title(title)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()
    return

def display_data_3d (data: np.ndarray, labels: np.ndarray, title = 'Plotted Data', \
    legend: bool = True, view_init: tuple = None) -> None:
    """
    Creates a plot for 3D multi-class classification data.
    
    :param data: The input dataset
    :param labels: The labels for the dataset
    :param legend: Whether to show a legend for the data (default: True)
    :param title: What the title for the plot will be
    :param view_init: The viewing elevation and azamuth angles in a tuple
    """
    # Reshapes the labels to make for easier comparison
    idx = labels.flatten()

    # Creates the plot
    fig = plt.figure()
    ax = plt.axes(projection ='3d')

    # Splits data by class
    label_count = np.unique(idx).shape[0]
    split_data = np.array([data[idx == i,:] for i in range(label_count)])

    # Plots the data
    color = cm.rainbow(np.linspace(0, 1, label_count))
    for i,c in zip(range(label_count), color):
        ax.scatter(split_data[i][:,0], split_data[i][:,1], split_data[i][:,2], \
            c=c.reshape(1,-1), label='Class {}'.format(i))
    
    if legend:
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')

    # Changes the viewing angle of the plot
    if view_init is not None:
        ax.view_init(view_init[0], view_init[1])

    plt.show()

def animate_data_3d (data: np.ndarray, labels: np.ndarray, filename: str, title = 'Plotted Data', \
    legend: bool = True) -> None:
    """
    Saves a video of a 3D plot of data being rotated.
    
    :param data: The input dataset
    :param labels: The labels for the dataset
    :param filename: The name (and location) of the created video
    :param legend: Whether to show a legend for the data (default: True)
    :param title: What the title for the plot will be
    """
    # Reshapes the labels to make for easier comparison
    idx = labels.flatten()

    # Creates figure
    fig = plt.figure()
    ax = plt.axes(projection ='3d')

    # Splits data by class
    label_count = np.unique(idx).shape[0]
    split_data = np.array([data[idx == i,:] for i in range(label_count)])

    if legend:
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')

    def init():
        # Initial plot
        color = cm.rainbow(np.linspace(0, 1, label_count))
        for i,c in zip(range(label_count), color):
            ax.scatter(split_data[i][:,0], split_data[i][:,1], split_data[i][:,2], \
                c=c.reshape(1,-1), label='Class {}'.format(i))
        return fig,

    def animate(i):
        ax.view_init(elev=-140, azim=i)
        return fig,

    anim = FuncAnimation(fig, animate, init_func=init, \
        frames=360, interval=20, blit=True)

    anim.save(filename)
    return