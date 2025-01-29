from jax import random, jit
import jax.numpy as jnp

@jit
def perturb_GOU(key, matrix, mu, theta, sigma, dt):
    """
    Perturb a matrix with a GOU process.

    Parameters
    ----------
    key: PRNGKey
        Random key
    matrix: ndarray
        The matrix to be perturbed
    mu: float
        The mean of the GOU process
    theta: float
        The mean reversion rate
    sigma: float
        The volatility/variance of the GOU process
    dt: float
        The time step for the simulation

    Returns
    -------
    ndarray
        The perturbed matrix
    """
    eta = random.normal(key, shape=matrix.shape)
    P = -theta * (matrix - mu) * dt + sigma * eta * matrix * jnp.sqrt(dt)
    # second order approximation
    #P = -theta * (matrix - mu) * dt + sigma * eta * matrix * jnp.sqrt(dt) + 0.5 * sigma**2 * matrix * (eta**2 *dt - dt)
    return P

def time_evolution_GOU(key, matrix, mu, theta, sigma, tau, dt):
    """
    Simulate the time evolution of a matrix perturbed by a GOU process for a given time `tau`.

    Parameters
    ----------
    key: PRNGKey
        Random key
    matrix: ndarray
        The matrix to be perturbed
    mu: float
        The mean of the GOU process
    theta: float
        The mean reversion rate
    sigma: float
        The volatility/variance of the GOU process
    tau: float
        The total time for the simulation
    dt: float
        The time step for the simulation

    Returns
    -------
    ndarray
        The perturbed matrix after the time evolution
    """
    n_steps = int(tau / dt)

    for _ in range(n_steps):
        key, sim_key = random.split(key)
        matrix += perturb_GOU(sim_key, matrix, mu, theta, sigma, dt)

    return matrix

def create_binary_dataset(rng, n_samples, input_dim, output_dim=1):
    """
    Create a binary dataset with random inputs and outputs.

    Parameters
    ----------
    rng: PRNGKey
        Random key
    n_samples: int
        Number of samples
    input_dim: int
        Dimension of the input
    output_dim:
        Dimension of the output

    Returns
    -------
    ndarray
        The input data
    """
    rng, x_key, y_key = random.split(rng, 3)

    X_train = random.randint(x_key, (n_samples, input_dim), minval=0, maxval=2)
    y_train = random.randint(y_key, (n_samples, output_dim), minval=0, maxval=2)

    return X_train, y_train


def simulate_perturbation_only(rng, w_perturb, n_steps, mu, theta, sigma, dt):
    """
    Simulate the perturbation of a matrix by a GOU process for a given number of steps and return the whole history.

    Parameters
    ----------
    rng: PRNGKey

    w_perturb: ndarray
        The initial matrix to be perturbed
    n_steps:
        Number of steps for the simulation
    mu: float
        The mean of the GOU process
    theta: float
        The mean reversion rate
    sigma: float
        The volatility/variance of the GOU process
    dt: float
        The time step for the simulation

    Returns
    -------
    ndarray
        The history of the perturbed matrix
    """
    weight_list = [w_perturb]
    for _ in range(n_steps):
        rng, gou_key = random.split(rng)
        w_perturb += perturb_GOU(gou_key, w_perturb, mu=mu, theta=theta, sigma=sigma, dt=dt)
        #set negative weights to zero
        w_perturb = jnp.where(w_perturb < 0, 0, w_perturb)
        weight_list.append(w_perturb)

    return jnp.array(weight_list)


def mu_LN_from_params(mu, sigma, theta, dt=0.01, tau= 1):
    '''
    Theoretical formula for the mean of the log-normal distribution of the GOU process
    '''

    return jnp.log(mu * jnp.sqrt(1 - sigma**2/2*theta))

def sigma_LN_from_params(mu, sigma, theta, dt=0.01, tau = 1):
    '''
    Theoretical formula for the standard deviation of the log-normal distribution of the GOU process
    '''

    return jnp.sqrt(jnp.log(1/(1  - sigma**2/(2*theta))))

