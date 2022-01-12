import numpy as np

def PCA(X: np.ndarray, threshold: float = None, num_dim: int = None) -> np.ndarray:
    """
    Uses Principal Component Analysis to reduce dimensionality of dataset X usually either
    a threshold variance OR the number of dimensions.
    X: The dataset to be transformed, rows being datapoints and columns are features
    threshold: The explained variance [0,1] that is retained in the reduced dimensions.
        ex. 0.95 will return a reduced dataset where the eigenvectors still account for at
        least 95% of the variance in the data. 
    num_dim: The number of dimensions to reduce X into.
    """
    if threshold is None and num_dim is None:
        raise TypeError("Must specify a variance threshold or the number of dimensions.")
    if threshold is not None and num_dim is not None:
        raise TypeError("Cannot specify both a variance threshold and number of dimensions.")
    # Mean-centers the data
    X_centered = X - X.mean(axis = 0)
    # Gets the covariance matrix
    X_cov = np.cov(X_centered, rowvar=False)
    # eigh works faster for symmetric matrices
    eig_values, eig_vectors = np.linalg.eigh(X_cov)

    # The values are already sorted in ascending order, needs to be descending
    eig_values = eig_values[::-1]
    eig_vectors = eig_vectors[:,::-1]

    # Threshold
    if threshold is not None:
        new_dim = 0
        sum_eigv = np.sum(eig_values)
        # Gets the explained variance
        explained_variances = list(map(lambda x: x / sum_eigv, eig_values))
        th = 0
        for i in range(len(eig_values)):
            th += explained_variances[i]
            if th >= threshold:
                new_dim = i+1
                break
    # Given number of dimensions
    else:
        new_dim = num_dim

    # Reduces the eigenvectors
    reduced_eig_vectors = eig_vectors[:,:new_dim]
    # Calculates the reduced dataset
    X_reduced = (reduced_eig_vectors.T @ X_centered.T).T

    return X_reduced

if __name__ == '__main__':
    import pandas as pd

    # Reads the iris dataset
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
    # Gets the pure data and converts to numpy array
    df_np = df.iloc[:,0:4].to_numpy()

    # Performs PCA
    reduced_df_np = PCA(df_np, threshold=0.95)
    #reduced_df_np = PCA(df_np, num_dim=2)

    print('New dimension count: {}'.format(reduced_df_np.shape[1]))

    

