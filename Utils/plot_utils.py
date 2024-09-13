import matplotlib.pyplot as plt
import jax.numpy as jnp


def plot_weights(w_mat, title="weights", bins=100, save_path=None, show=False):
    """
    Plot the weights in a histogram.

    Parameters
    ----------
    w_mat : ndarray
        The weight matrix to be plotted. This matrix will be flattened
        before plotting.
    title : str, optional
        The title of the plots. Default is "weights".
    bins : int, optional
        The number of bins to use in the histograms. Default is 100.
    save_path : str, optional
        The file path to save the figure. If None, the figure will not be saved. Default is None.

    Returns
    -------
    None
        This function does not return any values. It displays and optionally
        saves the histogram plots of the weights.
    """
    w_mat = w_mat.flatten()
    # Plot the weights regularly and in log scale
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(w_mat, bins=bins, density=True)
    plt.title(title)

    plt.subplot(1, 2, 2)
    plt.hist(jnp.clip(jnp.log(w_mat), a_min = -1e6), bins=bins, density=True)
    plt.title(title + ' (log)')

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_weight_dynamics(w_list, title="Weight Dynamics", weights_to_show=100, log=False, save_path=None, show=False):
    """
    Plot the dynamics of the weights over time.

    Parameters
    ----------
    w_list : ndarray
        A 2D array where each row represents the weights at a different
        time step.
    title : str, optional
        The title of the plot. Default is "Weight Dynamics".
    weights_to_show : int, optional
        The number of weights to show in the plot. Default is 100.
    log : bool, optional
        If True, the logarithm of the weights will be plotted. Default is False.
    save_path : str, optional
        The file path to save the figure. If None, the figure will not be saved. Default is None.

    Returns
    -------
    None
        This function does not return any values. It displays and optionally
        saves the dynamics of the weights over time.
    """
    w_list = w_list[:, :weights_to_show]
    if log:
        w_list = jnp.log(w_list)
        title = title + ' (log)'
    plt.figure(figsize=(12, 4))
    plt.plot(w_list)
    plt.title(title)

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


def plot_eigenvalues(w_init, w_final, bias=0.0, log=False, save_path=None, show=False):
    """
    Plot the eigenvalues of the initial, final, and difference of weights.

    Parameters
    ----------
    w_init : ndarray
        The initial weight matrix.
    w_final : ndarray
        The final weight matrix.
    bias : float, optional
        A bias term to subtract from the weights before computing eigenvalues.
        Default is 0.0.
    log : bool, optional
        If True, the eigenvalues will be plotted on a logarithmic scale. Default
        is False.
    save_path : str, optional
        The file path to save the figure. If None, the figure will not be saved. Default is None.

    Returns
    -------
    None
        This function does not return any values. It displays and optionally
        saves the plots of the eigenvalues.
    """
    delta = w_final - w_init

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(jnp.absolute(jnp.linalg.eigvals(w_init - bias)))
    # Log scale
    if log:
        plt.yscale('log')
    plt.title('Eigenvalues of initial weights')

    plt.subplot(1, 3, 2)
    plt.plot(jnp.absolute(jnp.linalg.eigvals(w_final - bias)))
    if log:
        plt.yscale('log')
    plt.title('Eigenvalues of final weights')

    plt.subplot(1, 3, 3)
    plt.plot(jnp.absolute(jnp.linalg.eigvals(delta)))
    if log:
        plt.yscale('log')
    plt.title('Eigenvalues of weight variation')

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()

