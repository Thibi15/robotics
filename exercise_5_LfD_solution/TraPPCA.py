import numpy as np
import scipy.interpolate as sp

class TraPPCA:
    """
    A class that learns a model using TraPPCA. Related sources:
    
    E. Aertbeliën and J. De Schutter, “Learning a predictive model of human gait 
    for the control of a lower-limb exoskeleton,” in 5th IEEE RAS/EMBS International 
    Conference on Biomedical Robotics and Biomechatronics, 2014, pp. 520 - 525.
    
    C. Vergara Perico, J. De Schutter, and E. Aertbeliën, "Combining imitation learning 
    with constraint-based task specification and control", IEEE Robotics and Automation 
    Letters, vol. 4, no. 2, pp. 1892 - 1899, 2019.

    C. Vergara Perico, J. De Schutter, and E. Aertbeliën, "Learning robust manipulation 
    tasks involving contact using trajectory parameterized probabilistic principal 
    component analysis", in IEEE/RSJ International Conference on Intelligent Robots 
    and Systems (IROS), 2020, pp. 8336 - 8343.

    T. Callens, A. van der Have, S. Van Rossom, J. De Schutter, and E. Aertbeliën, 
    “A framework for recognition and prediction of human motions in human-robot 
    collaboration using probabilistic motion models,” IEEE Robotics and Automation 
    Letters (RAL) paper presented at the 2020 IEEE/RSJ International Conference on 
    Intelligent Robots and Systems (IROS), vol. 5, no. 4, pp. 5151 – 5158, 2020.
    """

    def __init__(self, training_data, modes=5):
        """ Initializes an object of the TraPPCA class
            Inputs:
            - training_data: N x M x L array containing the training data, with N the number of demonstrations, 
                             M the number of dimensions and L the number of samples
            - modes: number of modes (integer)
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

        # Number of modes
        self._modes = modes

        # Model parameters
        self._sigma = None
        self._x = None

        # Splines to evaluate h and b at different progress values (created during the
        # learning phase). h_spline and b_spline are lists of size dim_number,
        # with a spline for every dimension.
        self._h_spline = None
        self._b_spline = None

        # Conditioned model parameters
        self._sigma_cond = None
        self._x_cond = None
        self._cond_meas_noise = 0.001

        # Initialize via-point list
        self._via_points = []

        # Measurement noise variance (gets calculated during training phase)
        self._meas_noise = None

    def train(self):
        """ Trains the model using the TraPPCA algorithm.
            Inputs: None
            Outputs: None
        """
        
        # Get values and data
        training_data = self._training_data
        demo_number = self._demo_number
        traj_length = self._traj_length
        modes = self._modes
        d = self._dim_number*traj_length

        # step 1: put all trials after each other
        sample_matrix = np.empty((0, d))
        for i in range(0, demo_number):
            demo = np.array([[x for dimension in training_data[i] for x in dimension]])
            sample_matrix = np.append(sample_matrix, demo, axis=0)

        # step 2: create sample covariance matrix
        sample_cov_matrix = np.cov(sample_matrix.T)

        # step 3: calculate eigenvectors/eigenvalues
        eigvals, eigvecs = np.linalg.eig(sample_cov_matrix)
        eigvals = np.sort(eigvals)
        eigvals = np.flip(eigvals)

        # Keep only the real part
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)

        # step 4: select principal components
        princ_vals = np.diag(eigvals[:modes])
        princ_vecs = eigvecs[:, :modes]

        # step 5: calculate H-matrix, meas_noise and b (max. likelihood sol.)
        
        b_ml = np.mean(sample_matrix, axis=0)
        
        if np.sum(eigvals[modes:]) <= 0:
            if modes > 0:
                print("Retry training of TraPPCA model with less modes: modes = " + str(modes) + "-> modes = " + str(modes-1))
                self._modes = self._modes - 1
                self.train()

            else:
                print("Training of TraPPCA model failed")

        else:
            meas_noise = np.sqrt(1/(d-modes) * np.sum(eigvals[modes:]))  
            h_ml = np.dot(princ_vecs, np.sqrt(
                (princ_vals-np.power(meas_noise, 2) * np.eye(len(princ_vals)))))

            # Create splines to evaluate h and b at different progress values
            h_spline = self.create_spline_matrices(h_ml.T)
            b_spline = self.create_spline_matrices(b_ml)

            self._meas_noise = meas_noise  # This is a scalar value (!)
            self._x = np.zeros((modes))
            self._sigma = np.eye(len(self._x))

            self._h_spline = h_spline
            self._b_spline = b_spline

    def create_spline_matrices(self, matrix):
        """ Creates a matrix with splines as elements. 
            Inputs: 
            - matrix: 
                - Option 1: 1D array containing a time series, which can be split up in M trajectories
                            corresponding to the dimensions in the training data
                - Option 2: 2D array with each of the m rows containing a time series, which can be split up in M trajectories
                            corresponding to the dimensions in the training data.
            Outputs:
            - splines: 
                - Option 1: 1D array of length M, containing a spline for every dimension in the training data
                - Option 2: 2D M x m array, containing a spline at every element, corresponding to the dimensions in the training data
                            and the time series in the input matrix
        """

        dim_number = self._dim_number

        # If "matrix" is one-dimensional, create splines for the one dimension
        if len(np.shape(matrix)) == 1:

            splines = self.create_splines(matrix)
            return splines

        # If "matrix" is two-dimensional, create splines for each row
        elif len(np.shape(matrix)) == 2:

            splines = np.empty((dim_number, 0))

            # Create splines for each row in "matrix"
            for mode in matrix:

                mode_splines = self.create_splines(mode)
                splines = np.append(splines, mode_splines, axis=1)
            return splines

        else:
            raise ValueError("Matrix has more than 2 dimensions")
        
    def create_splines(self, sequence):
        """ Creates splines of 'sequence'. One spline is created for every dimension.
            Inputs: 
            - sequence: 1D array containing a time series, which can be split up in M trajectories
                        corresponding to the dimensions in the training data
            Outputs:
            - splines: 1D array of length M, containing a spline for every dimension in the training data
        """
        
        dim_number = self._dim_number
        length = self._traj_length
        progress = self._progress

        spline_list = np.array([[sp.UnivariateSpline(progress, sequence[i*length:
                                                     (i+1)*length])]
                                for i in range(0, dim_number)])

        return spline_list
    
    def evaluate_basisfunctions(self, x):
        """ Evaluates the value of all the basis functions at the given progress.
            Inputs:
            - x: progress input, which can be a single value (floating point) or a list of values (list/array)
            Output: 
            - h_eval: self._h_spline evaluated at x, resulting in an M N_x x N_modes array, with M the number of dimensions, 
                      N_x the number of evaluation points in x and N_modes the number of modes
            - b_eval: self._b_spline evaluated at x, resulting in an M N_x array, with M the number of dimensions and N_x
                      the number of evaluation points in X
        """

        # If only one evaluation point is given as input, x is a float.
        # However, for the rest of the calculations, it is easier if x is
        # always a list
        if not (isinstance(x, list) or isinstance(x, np.ndarray)):
            x = [x]

        modes = self._modes
        h_spline = self._h_spline
        b_spline = self._b_spline
        h_eval = np.empty((0, modes))

        # Loop over all splines in the h_spline matrix
        for dim in h_spline:
            dim_eval = np.array([mode(x) for mode in dim])
            h_eval = np.append(h_eval, dim_eval.T, axis=0)

        b_eval = np.array([k[0](x) for k in b_spline])
        b_eval = np.ndarray.flatten(b_eval)

        return h_eval, b_eval
    
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
        """ This function recalculates the weighting coefficients of the TraPPCA
            model. The conditioning, as with ProMPs, is equal to a simple Kalman
            filter.
            Inputs:
            - meas_noise_value: measurement noise (floating point)
            Outputs:
            - x_cond, sigma_cond: conditioned model parameters
                - x_cond: array of length N_modes, with N_modes the number of modes
                - sigma_cond: array of shape N_modes x N_modes, with N_modes the number of modes
        """
        
        # Initialize the conditioned model parameters with the most appropriate values
        if self._x_cond is None:
            x_cond = self._x
            sigma_cond = self._sigma

        else:
            x_cond = self._x_cond
            sigma_cond = self._sigma_cond

        # Iterate over the available via-points
        for viapoint in self._via_points:

            # Prediction step
            if meas_noise_value is None:
                meas_noise_value = self._cond_meas_noise                
            measurement_noise = np.eye(np.shape(sigma_cond)[0]) * meas_noise_value
            
            sigma_cond = sigma_cond + measurement_noise

            # Update step
            y = viapoint["value"]
            s = viapoint["progress"]
            sigma_y = viapoint["sigma"]

            progress = np.array([s])     

            # Evaluate basis functions
            h_viapoint, b_viapoint = self.evaluate_basisfunctions(progress)            

            # Conditioning
            inv = np.linalg.inv(
                sigma_y + np.dot(np.dot(h_viapoint, sigma_cond), h_viapoint.T))
            aux = np.dot(np.dot(sigma_cond, h_viapoint.T), inv)

            innovation = y - (np.dot(h_viapoint, x_cond) + b_viapoint)

            # Update model
            x_cond = x_cond + np.dot(aux, innovation)
            sigma_cond = sigma_cond - np.dot(np.dot(aux, h_viapoint), sigma_cond)

        self._x_cond = x_cond
        self._sigma_cond = sigma_cond

        # Delete used via-points, since the model is already conditioned on them
        self.clear_via_points()

        return x_cond, sigma_cond
    
    def is_conditioned(self):
        """ Returns True if the model is already conditioned on some via-points.
        """
        if self._x_cond is None:
            return False
        return True
    
    def clear_via_points(self):
        self._via_points = []

    def clear_cond(self):
        self._x_cond = None
        self._sigma_cond = None

    def reset_model(self):
        self.clear_via_points()
        self.clear_cond()
    
    @property
    def modes(self):
        return self._modes
    
    @property
    def dim(self):
        return self._dim_number
    
    @property
    def traj_length(self):
        return self._traj_length

    @property
    def x(self):
        return self._x

    @property
    def x_cond(self):
        return self._x_cond

    @property
    def sigma(self):
        return self._sigma
    
    @property
    def sigma_cond(self):
        return self._sigma_cond

    @property
    def meas_noise(self):
        return self._meas_noise
    
    @property
    def dprogress(self):
        return self._dprogress
    
    @property
    def via_points(self):
        return self._via_points
    

    
