from jax import random
import jax.numpy as jnp


def perturb_GOU(key, matrix, mu, theta, sigma, dt):
    eta = random.normal(key, shape=matrix.shape)
    P = -theta * (matrix - mu) * dt + sigma * eta * matrix * jnp.sqrt(dt)

    return P


def create_binary_dataset(rng, n_samples, input_dim, output_dim=1):
    rng, x_key, y_key = random.split(rng, 3)

    X_train = random.randint(x_key, (n_samples, input_dim), minval=0, maxval=2)
    y_train = random.randint(y_key, (n_samples, output_dim), minval=0, maxval=2)

    return X_train, y_train


def simulate_perturbation_only(rng, w_perturb, n_steps, mu, theta, sigma, dt):
    weight_list = [w_perturb]
    for _ in range(n_steps):
        rng, gou_key = random.split(rng)
        w_perturb += perturb_GOU(gou_key, w_perturb, mu=mu, theta=theta, sigma=sigma, dt=dt)
        weight_list.append(w_perturb)

    return jnp.array(weight_list)


def mu_LN_from_params(mu, sigma, theta, dt):
    return jnp.log(mu * jnp.sqrt(1 - sigma**2/2*theta))

def sigma_LN_from_params(mu, sigma, theta, dt):
    return jnp.sqrt(jnp.log(1/(1  - sigma**2/(2*theta))))

