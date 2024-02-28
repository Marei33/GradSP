
import pandas as pd
import tensorflow as tf
import numpy as np
import os as os
import matplotlib.pyplot as plt

# own code base
import sys
sys.path.append("..")
import tf_loss_functions as lf
import splines as sp


#####################
# Parameter Setting #
#####################

# set working directory
os.chdir(os.path.dirname(__file__))
dirname = os.path.dirname(__file__) 

# set spline hyperparameters
basis_dimension = 20        # number of B-spline basis functions
degree_bsplines = 3         # degree of B-splines
penalty_diff_order = 2      # order of difference penalty

# set optimization parameters
weights_init_mean = 0.0
weights_init_sd = 0.1
lambda_param_init = tf.math.log(10e0) 
num_epochs_lambda_param = 10000 
num_epochs_splines = 1
learning_rate_lambda_param = 1
learning_rate_splines = 1

method = "gcv"

np.random.seed(33)


####################
# Data Preparation #
####################



n_sim = 400000
x_dim = np.arange(0, 1, 0.001)
sd_sim = 0.2


x1 = np.array(pd.read_csv(f"df_x1.csv", index_col=0).iloc[:,0])
y = np.array(pd.read_csv(f"df_y.csv", index_col=0).iloc[:,0])


x1_ps = sp.pspline(
    x=x1, 
    degree_bsplines=degree_bsplines, 
    penalty_diff_order=penalty_diff_order, 
    knot_type="equi", 
    basis_dimension=basis_dimension,
    )


labels = np.expand_dims(y, 1)



##################
# Model Building #
##################

# set parameters for optimization
initializer = tf.keras.initializers.TruncatedNormal(mean=weights_init_mean, stddev=weights_init_sd, seed=13)
opt_lambda_param = tf.keras.optimizers.Adam(learning_rate=learning_rate_lambda_param)
opt_splines = tf.keras.optimizers.Adam(learning_rate=learning_rate_splines)

lambda_param = tf.Variable(lambda_param_init, name="lambda_param", dtype=tf.float32, trainable=True)
weights = tf.Variable(initializer(shape=(basis_dimension, 1)), name="weights")

epochs_saved = []
lambda_param_est_values = []
pen_loss_values = []


# define function to do the optimization in batches
def batch_data(data, labels, batch_size, shuffle=True):

    # shuffle the data
    if shuffle:
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
    
    num_samples = len(data)
    num_batches = num_samples // batch_size
    batches = []
    
    for i in range(num_batches):
        batch_data = data[i * batch_size : (i + 1) * batch_size]
        batch_labels = labels[i * batch_size : (i + 1) * batch_size]
        batches.append((batch_data, batch_labels))
    
    if num_samples % batch_size != 0:
        # Add the remaining samples as a smaller batch
        batch_data = data[num_batches * batch_size:]
        batch_labels = labels[num_batches * batch_size:]
        batches.append((batch_data, batch_labels))
    
    return batches




#####################
# Optimization Loop #
#####################


# Loop for optimization
for i in range(num_epochs_lambda_param):
    batches = batch_data(x1, labels, batch_size=400, shuffle=True)
    for x1_batch, labels_batch in batches:
        # set up spline
        x1_batch_ps = sp.pspline(
            x=x1_batch, 
            degree_bsplines=degree_bsplines, 
            penalty_diff_order=penalty_diff_order, 
            knot_type="equi", 
            basis_dimension=basis_dimension,
            )

        # compute loss and optimize
        if method == "gcv":
            loss = lambda: lf.gcv_1d(y = labels_batch, 
                                    design_matrix_Z = x1_batch_ps.design_matrix_d,
                                    reg_matrix_K = x1_batch_ps.penalty_matrix_d, 
                                    reg_param = tf.exp(lambda_param)
                                    )
        if method == "reml":
            loss = lambda: lf.reml_1d(y = labels_batch, 
                                    design_matrix_Z = x1_batch_ps.design_matrix_d,
                                    reg_matrix_K = x1_batch_ps.penalty_matrix_d, 
                                    reg_param = tf.exp(lambda_param)
                                    )
        opt_lambda_param.minimize(loss, var_list=[lambda_param])
        pen_loss = loss().numpy().item()

        for h in range(num_epochs_splines):
            loss = lambda: lf.penalized_least_squares(y = labels_batch, 
                                                    weights = weights, 
                                                    design_matrix = x1_batch_ps.design_matrix_d, 
                                                    reg_param = tf.exp(lambda_param), 
                                                    penalty_matrix = x1_batch_ps.penalty_matrix_d
                                                    )
            opt_splines.minimize(loss, var_list=[weights])

    if (i+1)%100 == 0: # save every 100th epoch
        print(f"Epoch: {i+1}") 
        epochs_saved.append(i+1)
        lambda_param_est = tf.exp(lambda_param).numpy()
        lambda_param_est_values.append(lambda_param_est)
        pen_loss_values.append(pen_loss)
        weights_est = weights.numpy()

        filename = f"spline_method={method}_epoch={i+1}.npz"
        np.savez(file = filename, reg_param = lambda_param_est, weights = weights_est)

        training_frame = pd.DataFrame(data=np.array([epochs_saved, lambda_param_est_values, pen_loss_values]).T, columns=["Epoch", "Lambda Parameter", f"{method}"])
        training_frame.to_csv(f"smoothing_param_method={method}_epochs={num_epochs_lambda_param}.csv")

print(f"Optimization finished")


# save final results
lambda_param_est = tf.exp(lambda_param).numpy()
weights_est = np.dot(x1_ps.U, weights.numpy())


filename = f"spline_method={method}_epochs={num_epochs_lambda_param*num_epochs_splines}.npz"
np.savez(file = filename, reg_param = lambda_param_est, weights = weights_est)

training_frame = pd.DataFrame(data=np.array([epochs_saved, lambda_param_est_values, pen_loss_values]).T, columns=["Epoch", "Lambda Parameter", f"{method}"])
training_frame.to_csv(f"smoothing_param_OPT_method={method}_epochs={num_epochs_lambda_param}.csv")

