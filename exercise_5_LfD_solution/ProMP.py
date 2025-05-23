import numpy as np
import scipy.interpolate as sp

class ProMP:
    """
    A class that learns a model using ProMP. Related sources:
    
    A. Paraschos, C. Daniel, J. R. Peters, and G. Neumann, “Probabilistic movement 
    primitives,” in Advances in Neural Information Processing Systems, vol. 26, 2013.

    A. Paraschos, C. Daniel, J. R. Peters, and G. Neumann, “Using probabilistic movement 
    primitives in robotics," Autonomous Robots, vol. 42, no. 3, pp. 529 – 551, 2018.
    """

    def __init__(self, training_data, n_weights_per_dim = 10):
        """ Initializes an object of the ProMP class
            Inputs:
            - training_data: N x M x L array containing the training data, with N the number of demonstrations, 
                             M the number of dimensions and L the number of samples
            - n_weights_per_dim: number of weights per dimension (integer)
            Outputs: None
        """

        self._training_data = training_data
        self._demo_number = len(self._training_data)
        self._dim_number = len(self._training_data[0])
        self._traj_length = len(self._training_data[0][0, :])

        # Sample progress interval
        self._dprogress = 1/(self._traj_length-1)

        # Compute progress
        self._progress = np.linspace(0,1,num=self._traj_length)

        # Number of weights (per dimension)
        self._n_weights_per_dim = n_weights_per_dim
        self._n_weights = self._dim_number * self._n_weights_per_dim

        # Centers for radial basis functions
        self.centers = np.linspace(0, 1, self._n_weights_per_dim)

        # Model parameters
        self._weight_mean = np.zeros(self._n_weights)
        self._weight_cov = np.eye(self._n_weights)

        # Conditioned model parameters
        self._weight_mean_cond = None
        self._weight_cov_cond = None
        self._cond_meas_noise = 0.001

        # Initialize via-point list
        self._via_points = []

    def radial_basis_functions(self, width = 3.5*10**(-3), normalize = True):
        """ Generates radial basis functions.
            Inputs: 
            - width: width of the radial basis functions (floating point)
            - normalize: indicates whether the radial basis functions for one dimension should be normalized  (boolean)
            Outputs: 
            - radial_full: N_w x M L array containing the radial basis functions for all dimensions, with N_w the number of weights, 
                           M the number of dimensions and L the number of samples
        """

        # Determine the radial basis functions for one dimension
        radial_1 = np.exp(-(np.atleast_2d(self._progress) - self.centers[:, np.newaxis]) ** 2 / (2.0 * width))

        # If required, normalize the radial basis functions for one dimension
        if normalize:
            radial_1 = radial_1/np.sum(radial_1)

        # Determine the radial basis functions for all dimensions
        radial_full = np.zeros((self._n_weights_per_dim * self._dim_number, len(self._progress) * self._dim_number))
        for j in range(self._dim_number):
            radial_full[self._n_weights_per_dim * j:self._n_weights_per_dim * (j + 1), len(self._progress) * j:len(self._progress) * (j + 1)] = radial_1

        return radial_full
    
    def train(self, lmbda = 1e-12, width = 3.5*10**(-3), normalize = True):
        """ Trains the model using the ProMP algorithm.
            Inputs:
            - lmbda: regression parameter (floating point)
            - width: width of the radial basis functions (floating point)
            - normalize: indicates whether the radial basis functions for one dimension should be normalized (boolean)
            Outputs:
            - self._radial_basis: N_w x M L array containing the radial basis functions for all dimensions, 
                                  with N_w the number of weights, M the number of dimensions and L the number of samples
            - weights: N_demo x N_w array containing the weights, with N_demo the number of demonstrations and N_w the 
                       number of weights
            - self._weight_mean: N_w array containing the mean of the weights over the demonstrations
            - self._weight_cov: N_w x N_w array containing the covariance of the weights over the demonstrations
        """

        # Determine the radial basis functions
        self._radial_basis = self.radial_basis_functions(width, normalize).T

        # Compute the weight vector for each demonstration
        weights = np.empty([self._demo_number, self._n_weights_per_dim*self._dim_number])
        for i in range(0, self._demo_number):
            weights[i] = np.dot(np.dot(np.linalg.pinv(np.dot(self._radial_basis.T, self._radial_basis) + lmbda * np.eye(self._radial_basis.shape[1])), self._radial_basis.T), self._training_data[i].ravel())

        # Calculate the mean weights
        self._weight_mean = np.sum(weights,0)/self._demo_number

        # Calculate the covariance of the weights
        self._weight_cov = np.zeros([self._n_weights_per_dim*self._dim_number, self._n_weights_per_dim*self._dim_number])
        for i in range(0,self._demo_number):
            self._weight_cov = self._weight_cov + np.dot((weights[i:i+1]-np.asarray([self._weight_mean])).T, (weights[i:i+1]-np.asarray([self._weight_mean])))
        self._weight_cov = self._weight_cov/self._demo_number

        return self._radial_basis, weights, self._weight_mean, self._weight_cov
        
    def append_via_points(self, viapoint):
        """ Appends the via-points for conditioning with the provided input
            Inputs:
            - viapoint: dictionary containing keys "progress", "value" and "sigma". 
            - key "progress": progress value (floating point)
            - key "value": values for each dimension of the via-point (list, even if via-point is only one-dimensional)
            - key "sigma": covariance matrix of the via-point (array). If "sigma" is not passed, a standard value is used.
            Outputs: None
        """

        if viapoint.get("sigma") is None:
            viapoint["sigma"] = np.eye(len(viapoint["value"])) * 10 ** (-16)
        elif np.count_nonzero(viapoint.get("sigma") - np.diag(np.diagonal(viapoint.get("sigma")))) != 0:
            print("Sigma is not a diagonal matrix! The provided sigma is replaced by a standard matrix")
            viapoint["sigma"] = np.eye(len(viapoint["value"])) * 10 ** (-16)

        self._via_points.append(viapoint)
    
    def condition_on_viapoints(self, meas_noise_value = None):
        """ This function conditions the ProMP model on the provided via-points, corresponding to a simple Kalman filter.
            Inputs:
            - meas_noise_value: measurement noise (floating point)
            Outputs:
            - mean_cond, cov_cond: conditioned model parameters
                - mean_cond: array of length N_weights, with N_weights the number of weights
                - cov_cond: array of shape N_weights x N_weights, with N_weights the number of weights
        """
        
        # Initialize the conditioned model parameters with the most appropriate values
        if self._weight_mean_cond is None:
            mean_cond = self._weight_mean
            cov_cond = self._weight_cov

        else:
            mean_cond = self._weight_mean_cond
            cov_cond = self._weight_cov_cond

        # Iterate over the available via-points
        for viapoint in self._via_points:

            # Prediction step
            if meas_noise_value is None:
                meas_noise_value = self._cond_meas_noise                
            measurement_noise = np.eye(np.shape(cov_cond)[0]) * meas_noise_value
            
            cov_cond = cov_cond + measurement_noise

            # Update step
            y = viapoint["value"]
            s = viapoint["progress"]
            sigma_y = viapoint["sigma"]

            # Evaluate basis functions
            basis_functions = np.asarray([self._radial_basis[i,:] for i in range(0,len(self._radial_basis)) if (i+1)%self._traj_length == 0])

            # Conditioning
            inv = np.linalg.inv(
                sigma_y + np.dot(np.dot(basis_functions, cov_cond), basis_functions.T))
            aux = np.dot(np.dot(cov_cond, basis_functions.T), inv)

            innovation = y - np.dot(basis_functions, mean_cond)

            # Update model
            mean_cond = mean_cond + np.dot(aux, innovation)
            cov_cond = cov_cond - np.dot(np.dot(aux, basis_functions), cov_cond)

        self._weight_mean_cond = mean_cond
        self._weight_cov_cond = cov_cond

        # Delete used via-points, since the model is already conditioned on them
        self.clear_via_points()

        return mean_cond, cov_cond

    def is_conditioned(self):
        """ Returns True if the model is already conditioned on some via-points.
        """
        if self._weight_mean_cond is None:
            return False
        return True
    
    def clear_via_points(self):
        self._via_points = []

    def clear_cond(self):
        self._weight_mean_cond = None
        self._weight_cov_cond = None

    def reset_model(self):
        self.clear_via_points()
        self.clear_cond()

    @property
    def dim(self):
        return self._dim_number
    
    @property
    def traj_length(self):
        return self._traj_length
        
    @property
    def weight_mean(self):
        return self._weight_mean

    @property
    def weight_mean_cond(self):
        return self._weight_mean_cond

    @property
    def weight_cov(self):
        return self._weight_cov
    
    @property
    def weight_cov_cond(self):
        return self._weight_cov_cond    
    
    @property
    def via_points(self):
        return self._via_points
        
