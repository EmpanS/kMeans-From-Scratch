# Import useful libraries
import numpy as np

class KMeans:
    AVAILABLE_DIST_F = ["L1_norm", "L2_norm"]

    """A class used to perform the k-means clustering algorithm on a data set. The maximum number of
     iterations is set by the user, if it converges to a solution, it stops iterating. 

    Constants
    ----------
    AVAILABLE_DIST_F : (list of str)
        Contains the available distance functions. L1_norm is the Manhattan distance, L2_norm is the
        ordinary Euclidean distance.
    
    Attributes
    ----------
    k : (int)
        Represents the number of clusters
        
    X : (numpy array)
        The data to cluster, must be an (m x n)-numpy array with m observations and n features.
        
    verbose : (boolean)
        A boolean representing if printing should be done while training the model.
        
    h_params : (dictionary)
        Contains two hyper-parameters, number of iterations (n_iter) and distance function (dist_f)
        
    random_state : (int)
        Optional setting for the random state. The k-means algorithm does not guarantee finding a 
        global minimum, but the final clusters depends on the initial random cluters.

    labels : (numpy array)
        Contains the predicted label for each observation, i.e., what cluster it belongs to.

    cluster_centers (numpy array)
        Contains the n-dimensional coordinates for each cluster.

    Methods
    -------
    update_h_params(self, h_params)
        Updates hyper parameters.

    fit(self, X=None)
        Performs the k-means algorithm on the passed data X or, if no data is passed, on self.X 
        
    _calculate_distances(X, centers)
        Calculates the distances between all observations in X and all centers of clusters. Uses the
        distance function already specified as a hyper-parameter.    
        
    _validate_param(h_param, setting)
        Validate new hyper-parameter settings.
    """
    
    def __init__(self, k, X, verbose=True, h_params=None, random_state=None):
        self.k = k
        self.X = X
        self.verbose = verbose
        self.h_params = {'n_iter':100, 'dist_f':'L2_norm'}
        self.random_state=random_state

        if h_params != None:
            self.update_h_params(h_params)

        self.labels = np.full((X.shape[0], 1), np.nan)
        self.cluster_centers = np.full((self.k, 1), np.nan)
    
    def update_h_params(self, h_params):
        """Updates the hyper parameters.
        
        Parameters
        ----------
        h_params : (dict)
            Dictionary containing the hyper parameter/s and its updated setting/s.
        
        Returns
        -------
        None
        """
        
        if type(h_params) != dict:
            raise TypeError('The argument must be a dictionary.')
        for h_param, setting in h_params.items():
            self._validate_param(h_param, setting)
            self.h_params[h_param] = setting 
            
    def fit(self, X=None):
        """Performs the k-means algorithm. First, all observations in X gets randomly assigned a 
        label. Then, the function iterates until a solution is found (converged) or the maximum 
        number of iterations is reached. Each iteration performs the following:
            - Update labels
            - Check convergence
            - Calculate new cluster centers
            
        Parameters
        ----------
        X : (numpy array)
            The data to cluster, must be an (m x n)-numpy array with m observations and n features.
        
        Returns
        -------
        wss : (numpy array)
            A numpy array that saves the within cluster sum of squares for each iteration.    

        Labels : (numpy array)
            A numpy array containing all new labels for the observations in X.
        """
        
        if X == None:
            X = self.X
     
        if (n := self.random_state) != None:
            np.random.seed(n)

        # Initiate array to save within cluster sum of squares
        wss = np.zeros((1, self.h_params['n_iter']))

        # Randomly draw k observations and set them as the initial cluster centers 
        center_index = np.random.choice(X.shape[0], size=self.k, replace=False)
        cluster_centers = X[center_index]
        old_labels = None

        for iter in range(self.h_params['n_iter']):
            # Label the observations using the updated cluster centers
            distances  = self._calculate_distances(X, cluster_centers)
            labels = np.argmin(distances, axis=1)

            # Calculate the within-sum-of-squares
            wss[0,iter] = sum(np.min(distances, axis=1))

            # Check convergence
            if np.all(labels == old_labels):
                if self.verbose:
                    print(f"Converged to a solution after {iter} iterations!")
                return(wss[0,:(iter)], labels)
            else: 
                old_labels = labels

            # Calculate new cluster centers
            for i in range(self.k):
                cluster_centers[i] = np.sum(X[labels==i],axis=0)/(X[labels==i].shape[0])

        if self.verbose:
            print(f"Did not converged, reached max iterations. Completed {iter+1} iterations.")
        return(wss[0,:], labels)        
            
    def _calculate_distances(self, X, centers):
        """
        Calculates the distances between all observations in X and all cluster centers. The already
        specified distance function (found in self.h_params) is used to calculate the distances.
        
        Parameters
        ----------
        X : (numpy array)
            A matrix (m x n) containing all observations.
        
        centers : (numpy array)
            A matrix (k x n) where k is the number of clusters, containing all cluster centers. 
        
        Returns
        -------
        labels : (numpy array)
            A numpy array containing all new labels for the observations in X.
        """
        
        # Initiate a distance matrix
        distance_m = np.tile(centers.flatten(), (X.shape[0],1))

        # Duplicate data matrix to same dimension as distance matrix
        X_m = np.tile(X, (centers.shape[0]))
        
        if self.h_params["dist_f"] == "L2_norm":
            # Complete the distance matrix using the L2-norm
            distance_m = np.reshape(distance_m - X_m, (X.shape[0]*centers.shape[0], X.shape[1]))
            distance_m = np.sum(np.square(distance_m),axis=1, keepdims=True)
            
            # Reshape distance matrix
            distance_m = np.sqrt(np.reshape(distance_m, (X.shape[0], len(centers))))
            return(distance_m)
            
        elif self.h_params["dist_f"] == "L1_norm":
            # Complete the distance matrix using the L1-norm
            distance_m = np.reshape(distance_m - X_m, (X.shape[0]*centers.shape[0], X.shape[1]))
            distance_m = np.sum(np.abs(distance_m),axis=1, keepdims=True)
            
            # Reshape distance matrix
            distance_m = np.reshape(distance_m, (X.shape[0], len(centers)))
            return(distance_m)
        
        else:
            raise ValueError('Could not calculate distance, no distance function found.')
    
    def _validate_param(self, h_param, setting):
        """
        Validates a given hyper-parameter update. The update must must have a valid key and value.
        
        Parameters
        ----------
        h_param : (str)
            The hyper parameter to update 
        
        setting : (int) or (str)
            The new setting of the hyper parameter
            
        Returns
        -------
        None - (Throws an error if not valid.)
        """

        if h_param not in self.h_params.keys():
            raise KeyError("No hyper parameter is named " + str(h_param) + ", it is a wrong value of key. Must be either 'n_iter' or 'dist_f'.")

        if h_param == "n_iter":
            if type(setting) != int or setting <= 0:
                raise ValueError("n_iter must be a positive integer which " + str(setting) + " is not.")
        else: # Setting for the distance function
            if setting not in self.AVAILABLE_DIST_F:
                raise ValueError(str(setting) + " is not an available distance function. Available functions are: " + str(self.AVAILABLE_DIST_F))