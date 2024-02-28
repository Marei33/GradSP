#%%

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt



#%%
###################
# Data Simulation #
###################

# Setting the parameters:
n_sim = 400
x_dim = np.arange(0, 1, 0.001)
sd_sim = 0.2
np.random.seed(11)


# Data generation function
def f(x, a=0, b=1, m=1, k=1):
    return 1/5*(a + b * x + m * ((np.cos(15*k*x)-1) + 4*np.exp(-(2*x)**10)*np.sin(12*k*x)))


#%%
### x and y normed between 1 and -1 ###
# test different values for k

a_values = np.array([0, 0, 0])
b_values = np.array([0, 1, 1])
k_values = np.array([0, 0, 1])

types = ["random", "linear", "wiggly"]

for j in range(len(k_values)):
    x_array = []
    y_array = []

    for i in range(10):
        x = np.random.uniform(low=0, high=1, size = n_sim)
        y = f(x, a=a_values[j], b=b_values[j], k=k_values[j])
        error = np.random.normal(loc = 0., scale = sd_sim, size = n_sim)
        y = y + error

        x_array.append(x)
        y_array.append(y)

    # save results in csv
    x_df = pd.DataFrame(x_array)
    x_df.to_csv(f"./df_x_{types[j]}.csv")

    y_df = pd.DataFrame(y_array)
    y_df.to_csv(f"./df_y_{types[j]}.csv")

    # Plot the generated data and save it
    fig, axs = plt.subplots(2, 5, figsize=(25,10), sharey=True)
    for i in range(10):
        if i < 5:
            axs[0, i].scatter(x_array[i], y_array[i], s=2)
            axs[0, i].plot(x_dim, f(x_dim, a=a_values[j], b=b_values[j], k=k_values[j]), color="red")
            axs[0, i].set_xlabel("x")
            axs[0, i].set_ylabel("y")
        else:
            axs[1, i-5].scatter(x_array[i], y_array[i], s=2)
            axs[1, i-5].plot(x_dim, f(x_dim, a=a_values[j], b=b_values[j], k=k_values[j]), color="red")
            axs[1, i-5].set_xlabel("x")
            axs[1, i-5].set_ylabel("y")
    fig.set_facecolor('white')
    fig.suptitle(f"Sampled Data, Type = {types[j]}")
    plt.savefig(f"./data_generation_{types[j]}.png")
    plt.show()


