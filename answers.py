
class Lorenz1996Model:
    """Implementation of the Lorenz 1996 model.
    
    Use the `tendency()` method to compute the model tendencies (i.e., dx/dt)
    and use the `forward()` method to apply an integration step forward in time,
    using a fourth order Runge--Kutta scheme.
    
    Attributes
    ----------
    Nx : int
        The number of variables in the model.
    F : float
        The model forcing.
    dt : float
        The model integration time step.
    """

    def __init__(self, Nx, F, dt):
        """Initialise the model."""
        self.Nx = Nx
        self.F = F
        self.dt = dt

    def tendency(self, x):
        """Compute the model tendencies dx/dt.
        
        The tendencies are computed by batch using
        `numpy` vectorisation.
        
        Parameters
        ----------
        x : np.ndarray, shape (..., Nx)
            Batch of input states.
            
        Returns
        -------
        dx_dt : np.ndarray, shape (..., Nx)
            Model tendencies computed at the input states.
        """
        # TODO: implement it!
        xp = np.roll(x, shift=-1, axis=-1)
        xmm = np.roll(x, shift=+2, axis=-1)
        xm = np.roll(x, shift=+1, axis=-1)
        return (xp - xmm)*xm - x + self.F

    def forward(self, x):
        """Apply an integration step forward in time.
        
        This method uses a fourth-order Runge--Kutta scheme:
        k1 <- dx/dt at x
        k2 <- dx/dt at x + dt/2*k1
        k3 <- dx/dt at x + dt/2*k2
        k4 <- dx/dt at x + dt*k3
        k <- (k1 + 2*k2 + 2*k3 + k4)/6
        x <- x + dt*k
        
        Parameters
        ----------
        x : np.ndarray, shape (..., Nx)
            Batch of input states.
            
        Returns
        -------
        integrated_x : np.ndarray, shape (..., Nx)
            The integrated states after one step.
        """
        # TODO: implement it!
        k1 = self.tendency(x)
        k2 = self.tendency(x+self.dt/2*k1)
        k3 = self.tendency(x+self.dt/2*k2)
        k4 = self.tendency(x+self.dt*k3)
        k = (k1 + 2*k2 + 2*k3 + k4)/6
        return x + self.dt*k

def perform_true_model_integration(Nt, Ne=1, seed=None):
    """Perform an integration in time using the true model.
    
    The initial state is a batch of random fields.
    
    Parameters
    ----------
    Nt : int
        The number of integration steps to perform.
    Ne : int
        The batch size.
    seed : int
        The random seed for the initialisation.
        
    Returns
    -------
    xr : np.ndarray, shape (Nt+1, Ne, Nx)
        The integrated batch of trajectories.
    """
    # define rng
    rng = np.random.default_rng(seed=seed)

    # allocate memory
    xt = np.zeros((Nt+1, Ne, true_model.Nx))

    # initialisation
    xt[0] = rng.normal(loc=3, scale=1, size=(Ne, true_model.Nx))
    
    # TODO: implement the model integration for Nt steps
    for t in trange(Nt, desc='model integration'):
        xt[t+1] = true_model.forward(xt[t])
    
    # return the trajectory
    return xt

def extract_input_output(xt):
    # TODO: extract x (input)
    x = xt[:-1]
    # TODO: extract y (output)
    y = xt[1:]
    # return input/output
    return (x, y)

def make_dense_network(seed, num_layers, num_nodes, activation):
    """Build a sequential neural network using dense layers.
    
    Parameters
    ----------
    seed : int
        The random seed.
    num_layers : int
        The number of hidden layers.
    num_nodes : int
        The number of nodes per hidden layer.
    activation : str
        The activation function for the hidden layers.
        
    Returns
    -------
    network : tf.keras.Sequential
    """
    # set seed
    tf.keras.utils.set_random_seed(seed=seed)
    # TODO: create a sequential network
    network = tf.keras.models.Sequential()
    # TODO: add the input layer
    network.add(tf.keras.Input(shape=(true_model.Nx,)))
    # TODO: add the hidden layers
    for i in range(num_layers):
        network.add(tf.keras.layers.Dense(num_nodes, activation=activation))
    # TODO: add the output layer
    network.add(tf.keras.layers.Dense(true_model.Nx))
    # compile the neural network
    network.compile(loss='mse', optimizer='adam')
    # print short summary
    network.summary()
    # return the network
    return network

def compute_trajectories(network):
    """Compute the forecast skill trajectories.
    
    Parameters
    ----------
    network : tf.keras.Model
        The model to evaluate.
        
    Returns
    -------
    xt : np.ndarray, shape (Nt, Ne, Nx)
        The trajectories.
    """
    # allocate memory
    (Nt, Ne, Nx) = xt_fs.shape
    xt = np.zeros((Nt, Ne, Nx))
    
    # initialisation
    xt[0] = xt_fs[0]
    
    # TODO: implement the neural network integration
    for t in trange(Nt-1, desc='surrogate model integration'):
        x_norm = normalise_x(xt[t])
        y_norm = network.predict(x_norm, batch_size=Ne, verbose=0)
        xt[t+1] = denormalise_y(y_norm)
        
    return xt

def make_convolutional_network(seed, num_layers, num_filters, kernel_size, activation):
    """Build a sequential neural network with convolutional layers.
    
    Parameters
    ----------
    seed : int
        The random seed.
    num_layers : int
        The number of hidden layers.
    num_filters : int
        The number of convolution filters per hidden layer.
    kernel_size : int
        The convolution kernel size for the hidden layer.
    activation : str
        The activation function for the hidden layers.
        
    Returns
    -------
    network : tf.keras.Sequential
    """
    # set seed
    tf.keras.utils.set_random_seed(seed=seed)
    # reshape layers
    reshape_input = tf.keras.layers.Reshape((true_model.Nx, 1))
    reshape_output = tf.keras.layers.Reshape((true_model.Nx,))
    # padding layer
    border = kernel_size//2
    def apply_padding(x):
        x_left = x[..., -border:, :]
        x_right = x[..., :border, :]
        return tf.concat([x_left, x, x_right], axis=-2)
    padding_layer = tf.keras.layers.Lambda(apply_padding)   
    # TODO: create a sequential network
    network = tf.keras.models.Sequential()
    # TODO: add the input layer
    network.add(tf.keras.Input(shape=(true_model.Nx,)))
    # TODO: add the reshape_input layer
    network.add(reshape_input)
    # TODO: add the hidden layers
    for i in range(num_layers):
        network.add(padding_layer)
        network.add(tf.keras.layers.Conv1D(num_filters, kernel_size, activation=activation))
    # TODO: add the output layer
    network.add(tf.keras.layers.Conv1D(1, 1))
    # TODO: add the reshape_output layer
    network.add(reshape_output)
    # compile the neural network
    network.compile(loss='mse', optimizer='adam')
    # print short summary
    network.summary()
    # return the network
    return network

