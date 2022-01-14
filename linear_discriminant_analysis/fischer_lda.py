import numpy as np

from utils.helper_functions import coerce_1d_array

class FischerLDA():
    def __init__(self, num_components) -> None:
        """Initializes the model parameters"""
        self.num_components = num_components

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> tuple:
        """Finds the optimal LDF mean and scatters for classification."""
        # Compatibility...
        X_t = coerce_1d_array(X_train, axis=0)
        #y_t = coerce_1d_array(y_train, axis=0)
        y_t = y_train

        # Gets the number of classes
        class_count = np.unique(y_t).shape[0]

        # Splits the data by class
        X_t_split = np.array([X_t[y_t == i] for i in range(class_count)])

        # Finds the class means and the overall mean
        class_means = np.array([np.mean(X_t_split[i],axis=0) for i in range(class_count)])
        overall_mean = class_means.mean(axis=0)

        # Computes the BCSM for all classes
        bcsm = self.between_class_scatter(class_means, overall_mean, class_count)

        # Computes the WCSM
        wcsm = self.within_class_scatter(X_t_split, class_means, class_count)

        # Computes inv(wcsm)*bcsm and gets the eigenvalues
        a_inv_b = np.linalg.inv(wcsm) @ bcsm
        # Gets the eigenvalues and vectors of A
        eigval, eigvec = np.linalg.eig(a_inv_b)
        eigval = eigval.real
        eigvec = eigvec.real
        # Gets the largest eigenvalues
        max_eigvals = np.sort(eigval)[-self.num_components:]
        # Gets their associated eigenvectors
        eig_idxs = [list(eigval).index(max_eigvals[i]) for i in range(self.num_components-1,-1,-1)]
        # Stacks the row eigenvectors together
        row_eig_vectors = tuple([np.array(eigvec[:, eig_idxs[i]]) for i in range(self.num_components)])
        h = np.row_stack(row_eig_vectors).T

        # Converts the class means and scatters to the fischer LDF space
        fischer_means = [h.T @ class_means[c] for c in range(class_count)]
        scatters = [self.class_scatter(X_t_split[c], class_means[c]) for c in range(class_count)]
        fischer_scatters = [(h.T @ scatters[c]) @ h for c in range(class_count)]

        # Saves function values for making predictions
        self.class_count = class_count
        self.h = h
        self.fischer_means = fischer_means
        self.fischer_scatters = fischer_scatters

        # Returns the eigenvectors, the Fischer LDF means, and Fischer LDF scatters
        return h, fischer_means, fischer_scatters

    def predict(self, input):
        """Predicts the label of the data given in input."""
        # Compatibility
        x = coerce_1d_array(input, axis=0)

        # Transforms the data into the Fischer space
        fischer_data = x @ self.h
        # Shifts the data by the fischer class mean
        classes_shift = np.array([fischer_data - self.fischer_means[c] for c in range(self.class_count)])

        # Calculates the mahalanobis distance for each datapoint in each class
        d = np.array([[(classes_shift[c,i].T @ self.fischer_scatters[c]) @ classes_shift[c,i] \
            for c in range(self.class_count)] for i in range(fischer_data.shape[0])])
        
        # Gets the class with the smallest mahal. distance for each datapoint
        chosen_classes = np.argmin(d, axis=1)
        return chosen_classes
    
    def between_class_scatter(self, class_means, overall_mean, class_count):
        """Computes the between class scatter matrix of the dataset"""
        # Shifts the means to be 0-centered
        means_subt = class_means - overall_mean
        # Calculate the dot product for each shifted class mean
        means_subt_dots = [coerce_1d_array(means_subt[i]) @ coerce_1d_array(means_subt[i]).T for i in range(class_count)]
        # The BCSM is just the sum of these
        bcsm = np.sum(means_subt_dots, axis=0)
        return bcsm
    
    def within_class_scatter(self, split_data, class_means, class_count):
        """Computes the within class scatter matrix of the dataset"""
        # Gets the scatter matrix for each class
        scatters = [self.class_scatter(split_data[i], class_means[i]) for i in range(class_count)]
        # Gets the within class scatter matrix
        wcsm = np.sum(scatters, axis=0)
        return wcsm

    def class_scatter(self, data, class_mean):
        """Computes a scatter matrix for a class' data"""
        data_count = data.shape[0]
        # Shifts the data to mean of 0
        data_subt = data - class_mean
        # Gets dot product for each datapoint
        data_subt_dots = [coerce_1d_array(data_subt[i]) @ coerce_1d_array(data_subt[i]).T for i in range(data_count)]
        # Sums to get class scatter
        class_scatter = np.sum(data_subt_dots,axis=0)
        return class_scatter
