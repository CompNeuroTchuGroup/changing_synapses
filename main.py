
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

from Utils.models import *
from Utils.simulation import *
from Utils.plot_utils import *

import time

import json

from copy import copy
import os

def run_simulation(experiment_path):
    print(experiment_path)
    param_file = os.path.join(experiment_path, 'simulation_parameters.json')
    # Load parameters from file
    with open(param_file) as f:
        params_dict = json.load(f)

    rng = random.key(params_dict["seed"])

    # Initialize the network

    mu_ln = mu_LN_from_params(mu=1, sigma=0.1, theta=0.5, dt=0.001)
    sigma_ln = sigma_LN_from_params(mu =1, sigma=0.1, theta=0.5, dt=0.001)

    rng, net_key = random.split(rng)
    params = init_mlp(net_key, mu_ln, sigma_ln, **params_dict["network_parameters"])
    W_h_init = copy(params['W_h'])

    # create the dataset
    rng, data_key = random.split(rng)
    n_samples = params_dict["dataset_parameters"]["n_samples"]
    X_train, y_train = create_binary_dataset(data_key, n_samples=n_samples,
                                             input_dim=params_dict["network_parameters"]["input_size"])

    # Train the network
    loss_list = []
    acc_list = []

    train_weights_list = []

    training_parameters = params_dict["training_parameters"]
    num_epochs = training_parameters["num_epochs"]
    learning_rate = training_parameters["learning_rate"]

    for epoch in range(num_epochs):
        start_time = time.time()
        for x, y in zip(X_train, y_train):
            grads = grad(loss_mlp)(params, x, y)
            params['W_i'] -= learning_rate * grads['W_i']
            params['W_h'] -= learning_rate * grads['W_h']
            params['W_o'] -= learning_rate * grads['W_o']

            rng, gou_key = random.split(rng)
            simulation_parameters = params_dict["simulation_parameters"]
            params['W_h'] += perturb_GOU(gou_key, params['W_h'], **simulation_parameters)

            train_weights_list.append(copy(params['W_h']))

        acc_list.append(accuracy_mlp(params, X_train, y_train))
        loss_list.append(loss_mlp(params, X_train, y_train))

        if epoch % 10 == 0:
            epoch_time = time.time() - start_time
            train_loss = loss_mlp(params, X_train, y_train)
            train_acc = accuracy_mlp(params, X_train, y_train)
            print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
            print("Training set loss {}".format(train_loss))
            print("Training set accuracy {}".format(train_acc))

    # plot the loss and accuracy in 2 subplots

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_list)
    plt.ylim([0, 3])
    plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(acc_list)
    plt.ylim([0, 1.1])
    plt.title('Accuracy')
    plt.savefig(os.path.join(experiment_path, '_loss_acc.png'))
    #plt.show()

    # store the weight list
    train_weights_list = np.array(train_weights_list).reshape(num_epochs * n_samples, -1)
    np.save(os.path.join(experiment_path,'_train_weights.npy'), train_weights_list)

    #save figures
    train_weights_list = np.array(train_weights_list).reshape(num_epochs * n_samples, -1)
    plot_weight_dynamics(train_weights_list, "Weight Dynamics", weights_to_show=500, save_path=os.path.join(experiment_path, 'weight_dynamics.png'))
    plot_weight_dynamics(train_weights_list, "Weight Dynamics", weights_to_show=500, log=True, save_path=os.path.join(experiment_path,'weight_dynamics_log.png'))


    delta = params['W_h'] - W_h_init

    plot_weights(params['W_h'], "Final Weights", save_path=os.path.join(experiment_path,'final_weights.png'))
    plot_weights(W_h_init, "Initial Weights", save_path=os.path.join(experiment_path,'initial_weights.png'))
    plot_weights(delta, "Weight Variation", save_path=os.path.join(experiment_path, 'weight_variation.png'))



if __name__ == '__main__':
    experiment_names = ['pure_learning', 'noise_only', 'drift_only', 'full']
    for names in experiment_names:
        print(f"Running experiment {names}")
        experiment_path = os.path.join('Results', names)
        run_simulation(experiment_path)
        print(f"Experiment {names} completed")


