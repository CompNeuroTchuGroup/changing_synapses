import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from Utils.simulation import perturb_GOU, time_evolution_GOU


@jit
def relu(x):
    """Rectified linear unit activation function"""
    return jnp.maximum(0, x)


@jit
def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + jnp.exp(-x))


@jit
def forward_mlp(params, x):
    """
    Forward pass through the network.


    Parameters
    ----------
    params: dict
        Dictionary containing the parameters of the network
    x: jnp.ndarray
        Input data

    Returns
    -------
    jnp.ndarray
        Output of the network
    """
    # Forward pass through the network
    x = jnp.dot(x, params['W_i'])
    x = sigmoid(x)
    x = jnp.dot(x, params['W_h']- params['bias'])  # this bias is FUNDAMENTAL
    x = sigmoid(x)
    x = jnp.dot(x, params['W_o'])
    return sigmoid(x)

@jit
def forward_elm(params, x):
    """
    Forward pass through the network.

    Parameters
    ----------
    params: dict
        Dictionary containing the parameters of the network
    x: jnp.ndarray
        Input data

    Returns
    -------
    jnp.ndarray
        Output of the network
    """
    # Forward pass through the network
    x = jnp.dot(x, params['W_i']- params['bias']) + params['b_i']
    x = sigmoid(x)
    x = jnp.dot(x, params['W_o']) + params['b_o']
    return sigmoid(x)


def init_elm(key, mean_w,sigma_w, input_size, hidden_size, output_size, bias=0.0):
    """
    Initialize the parameters of an extreme learning machine.

    Parameters
    ----------
    key: jax.random.PRNGKey
        Random key
    mean_w: float
        Mean of the gaussian generating the weights of the central layer
    input_size: int
        Input size
    hidden_size: int
        Hidden size
    output_size: int
        Output size
    bias: float
        Bias term

    Returns
    -------
    dict
        Dictionary containing the parameters of the network
    """
    key_i, key_o, key_b_i, key_b_o = random.split(key, 4)
    params = {}
    params['W_i'] = jnp.exp(random.normal(key_i, shape=(input_size, hidden_size)) * sigma_w + mean_w)
    params['W_o'] = random.normal(key_o, (hidden_size, output_size))
    params['b_i'] = random.normal(key_b_i, (hidden_size, ))
    params['b_o'] = random.normal(key_b_o, (output_size, ))
    params['bias'] = jnp.ones(hidden_size) * bias
    return params

def init_mlp(key,  mean_w, sigma_w, input_size, hidden_size, output_size, bias=0.0):
    """
    Initialize the parameters of a multi-layer perceptron. The c

    Parameters
    ----------
    key: jax.random.PRNGKey
        Random key
    mean_w: float
        Mean of the gaussian generating the weights of the central layer
    sigma_w: float
        Standard deviation of the gaussian generating the weights of the central layer
    input_size: int
        Input size
    hidden_size: int
        Hidden size
    output_size: int
        Output size
    bias: float
        Bias term

    Returns
    -------
    dict
        Dictionary containing the parameters of the network

    """
    key_i, key_h, key_o = random.split(key, 3)
    params = {}
    params['W_i'] = random.normal(key_i, (input_size, hidden_size))
    params['W_h'] = jnp.exp(random.normal(key_h, shape=(hidden_size, hidden_size)) * sigma_w + mean_w)
    params['W_o'] = random.normal(key_o, (hidden_size, output_size))

    params['bias'] = jnp.ones(hidden_size) * bias
    return params




# these should be all rewritten
#---------------------------------
# Define cross-entropy loss function
def loss_mlp(params, x, y):
    """
    Compute the cross-entropy loss of the network.
    Parameters
    ----------
    params: dict
        Dictionary containing the parameters of the network
    x: jnp.ndarray
        Input data
    y: jnp.ndarray
        Target data

    Returns
    -------
    jnp.ndarray
        Cross-entropy loss of the network
    """

    y_pred = forward_mlp(params, x)
    y_pred = jnp.clip(y_pred, 1e-7, 1 - 1e-7)
    return -jnp.mean(y * jnp.log(y_pred) + (1 - y) * jnp.log(1 - y_pred))

# Define accuracy metric
def accuracy_mlp(params, x, y):
    """
    Compute the accuracy of the network.
    Parameters
    ----------
    params: dict
    x: jnp.ndarray
        Input data
    y: jnp.ndarray
        Target data

    Returns
    -------
    jnp.ndarray
        Accuracy of the network
    """
    y_pred = forward_mlp(params, x) > 0.5
    return jnp.mean(y_pred == y)

def loss_elm(params, x, y):
    """
    Compute the cross-entropy loss of the network.
    Parameters
    ----------
    params: dict
        Dictionary containing the parameters of the network
    x: jnp.ndarray
        Input data
    y: jnp.ndarray
        Target data

    Returns
    -------
    jnp.ndarray
        Cross-entropy loss of the network
    """

    y_pred = forward_elm(params, x)
    y_pred = jnp.clip(y_pred, 1e-7, 1 - 1e-7)
    return -jnp.mean(y * jnp.log(y_pred) + (1 - y) * jnp.log(1 - y_pred))

def accuracy_elm(params, x, y):
    """
    Compute the accuracy of the network.
    Parameters
    ----------
    params: dict
    x: jnp.ndarray
        Input data
    y: jnp.ndarray
        Target data

    Returns
    -------
    jnp.ndarray
        Accuracy of the network
    """
    y_pred = forward_elm(params, x) > 0.5
    return jnp.mean(y_pred == y)




