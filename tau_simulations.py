import matplotlib.pyplot as plt
import numpy as np
from qtconsole.mainwindow import background

from Utils.models import *
from Utils.simulation import *
from Utils.plot_utils import *

import time

params_dict = {
    "dataset_parameters": {
        "n_samples": 200
    },
    "network_parameters": {
        "input_size": 64,
        "hidden_size": 128,
        "output_size": 1,
        "bias": 1,
    },
    "training_parameters": {
        "num_epochs": 100,
        "learning_rate": 0.01
    },
    "simulation_parameters": {
        "mu": 1,
        "sigma": 0.1,
        "theta": 0.02,
        "dt": 0.001,
        "tau": 0.005
    },
    "seed": 42
}

rng = random.key(params_dict["seed"])

mu_LN = mu_LN_from_params(**params_dict["simulation_parameters"])
sigma_LN = sigma_LN_from_params(**params_dict["simulation_parameters"])

rng, data_key = random.split(rng)

X_train, y_train = create_binary_dataset(data_key, n_samples=params_dict["dataset_parameters"]["n_samples"],
                                         input_dim=params_dict["network_parameters"]["input_size"])

tau_list = jnp.arange(0.00, 0.05, 0.005)
loss_tau = []
acc_tau = []

training_parameters = params_dict["training_parameters"]
num_epochs = training_parameters["num_epochs"]
learning_rate = training_parameters["learning_rate"]
simulation_parameters = params_dict["simulation_parameters"]

print('tau_list', tau_list)




for tau in tau_list:
    acc_exp = []
    loss_exp = []
    print("Tau: ", tau)
    for n in range(10):
        print("Experiment: ", n)
        simulation_parameters["tau"] = tau
        rng, net_key = random.split(rng)
        params = init_elm(net_key, mu_LN, sigma_LN, **params_dict["network_parameters"])
        # params = simulate_training(gou_key, params, tau, num_epochs, X_train, y_train, learning_rate, simulation_parameters)
        for epoch in range(num_epochs):
            start_time = time.time()
            for x, y in zip(X_train, y_train):
                rng, gou_key = random.split(rng)
                # perturb the weights of W_i
                params['W_i'] = time_evolution_GOU(gou_key, params['W_i'], **simulation_parameters)
                # params['W_i'] += perturb_GOU(gou_key, params['W_i'], simulation_parameters['mu'], simulation_parameters['theta'], simulation_parameters['sigma'], simulation_parameters['dt'])

                grads = grad(loss_elm)(params, x, y)
                params['W_i'] -= learning_rate * grads['W_i']
                params['W_o'] -= learning_rate * grads['W_o']
                params['b_i'] -= learning_rate * grads['b_i']
                params['b_o'] -= learning_rate * grads['b_o']

            acc_exp.append(accuracy_elm(params, X_train, y_train))
            loss_exp.append(loss_elm(params, X_train, y_train))

            if epoch % 10 == 0:
                epoch_time = time.time() - start_time
                train_loss = loss_elm(params, X_train, y_train)
                train_acc = accuracy_elm(params, X_train, y_train)
                print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
                print("Training set loss {}".format(train_loss))
                print("Training set accuracy {}".format(train_acc))

    acc_tau.append(acc_exp)
    loss_tau.append(loss_exp)
    print("mean acc: ", np.mean(acc_exp))
    print("mean loss: ", np.mean(loss_exp))

#convert to numpy arrays
acc_tau = np.array(acc_tau)
loss_tau = np.array(loss_tau)


#save the results
np.save('old_results/acc_tau_tot.npy', acc_tau)
np.save('old_results/loss_tau_tot.npy', loss_tau)

# Plotting
#create figure
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(tau_list, np.mean(acc_tau, axis=1), label='Accuracy')
ax[0].fill_between(tau_list, np.mean(acc_tau, axis=1) - np.std(acc_tau, axis=1),
                   np.mean(acc_tau, axis=1) + np.std(acc_tau, axis=1), alpha=0.3)
ax[0].set_xlabel('Tau')
ax[0].set_ylabel('Accuracy')

ax[1].plot(tau_list, np.mean(loss_tau, axis=1), label='Loss')
ax[1].fill_between(tau_list, np.mean(loss_tau, axis=1) - np.std(loss_tau, axis=1),
                   np.mean(loss_tau, axis=1) + np.std(loss_tau, axis=1), alpha=0.3)
ax[1].set_xlabel('Tau')
ax[1].set_ylabel('Loss')
plt.savefig('tau_simulations.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()


