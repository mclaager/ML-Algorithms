import numpy as np
import matplotlib.pyplot as plt

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

def display_data (data: np.ndarray, labels: np.ndarray) -> None:
    """
    Creates a plot for 2D binary classification data.
    
    :param data: The input dataset
    :param labels: The binary labels for the dataset
    """
    # Reshapes the labels to make for easier comparison
    idx = labels.flatten()
    # Seperate and plot the data according to label
    zero_data = data[idx == 0,:]
    plt.scatter(zero_data[:,0], zero_data[:,1], c='blue', label='Class 0')
    one_data = data[idx == 1,:]
    plt.scatter(one_data[:,0], one_data[:,1], c='red', label='Class 1')
    # Adds some descriptive elements and shows the plot
    plt.legend()
    plt.title('Plotted Data')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()
    return