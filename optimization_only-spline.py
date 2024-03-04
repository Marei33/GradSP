
import pandas as pd
import tensorflow as tf
import numpy as np
import os as os
import matplotlib.pyplot as plt

# own code base
import tf_loss_functions as lf
import splines as sp




#####################
# parameter setting #
#####################

# set working directory
os.chdir(os.path.dirname(__file__))
dirname = os.path.dirname(__file__) 

# set spline hyperparameters
basis_dimension = 20    # number of B-spline basis functions
degree_bsplines = 3     # degree of B-splines
penalty_diff_order = 2  # order of difference penalty



# set optimization parameters
weights_init_mean = 0.0
weights_init_sd = 0.1

num_epochs_splines = 10
learning_rate_splines = 0.1

method = "reml"         # reml or gcv
 


####################
# Data Preparation #
####################
#%%

k = "wiggly"
j = 0


# load generated data
data_x = pd.read_csv(f"./data/df_x_{k}.csv", index_col=0)
data_y = pd.read_csv(f"./data/df_y_{k}.csv", index_col=0)


pen_loss_values = []
epochs_saved = []

x1 = data_x.iloc[j]
x1_name = f"data-{k}_j{j}"

# set up pspline
x1_ps = sp.pspline(x = x1, 
                    degree_bsplines = degree_bsplines, 
                    penalty_diff_order = penalty_diff_order, 
                    knot_type="equi", 
                    basis_dimension = basis_dimension)


labels = data_y.iloc[j]
labels = np.float32((labels - labels.mean()) / labels.std())
labels = np.expand_dims(labels, 1)


# determine optimal lambda parameter
lambda_size = 100
lambda_param = np.logspace(-2, 3, lambda_size, dtype=np.float32)
reml = (pd.read_csv(f"./data/REML_values_{k}.csv", index_col=0).to_numpy())

lambda_param_value = lambda_param[np.argmin(reml[:,j])]




##################
# Model Building #
##################


# set parameters for optimization
initializer = tf.keras.initializers.TruncatedNormal(mean=weights_init_mean, stddev=weights_init_sd, seed=13)
opt_splines = tf.keras.optimizers.Adam(learning_rate=learning_rate_splines)

lambda_param = tf.Variable(lambda_param_value, name = "lambda_param", dtype=tf.float32, trainable = False)
weights = tf.Variable(initializer(shape=(basis_dimension, 1)), name="weights")

# define loss function
loss = lambda: lf.penalized_least_squares(y = labels, 
                                        weights = weights, 
                                        design_matrix = x1_ps.design_matrix_d, 
                                        reg_param = lambda_param, 
                                        penalty_matrix = x1_ps.penalty_matrix_d
                                        )


# save initalization
weight_est = np.dot(x1_ps.U, weights.numpy())
lambda_param_est = lambda_param.numpy()
loss_est = loss().numpy().item()
filename = f"1d-spline_{x1_name}_method={method}_basisdim={basis_dimension}_epoch=0.npz"
np.savez(file = os.path.join(dirname, 'Results', filename), reg_param = lambda_param_est, weights = weight_est, loss = loss_est)



#####################
# Optimization Loop #
#####################


for h in range(num_epochs_splines):
    loss = lambda: lf.penalized_least_squares(y = labels, 
                                            weights = weights, 
                                            design_matrix = x1_ps.design_matrix_d, 
                                            reg_param = lambda_param, 
                                            penalty_matrix = x1_ps.penalty_matrix_d
                                            )
    opt_splines.minimize(loss, var_list=[weights])

    if h < 15: # save weights all first 15 epochs
        print(f"Epoch: {h+1}") 
        print(lambda_param.numpy())
        weight_est = np.dot(x1_ps.U, weights.numpy())
        loss_est = loss().numpy().item()
        filename = f"1d-spline_{x1_name}_method={method}_basisdim={basis_dimension}_epoch={h+1}.npz"
        np.savez(file = os.path.join(dirname, 'Results', filename), reg_param = lambda_param_value, weights = weight_est, loss = loss_est)
    else:
        if (h+1)%10 == 0: # then save only every 10th epoch
            print(f"Epoch: {h+1}") 
            print(lambda_param.numpy())
            weight_est = np.dot(x1_ps.U, weights.numpy())
            loss_est = loss().numpy().item()
            filename = f"1d-spline_{x1_name}_method={method}_basisdim={basis_dimension}_epoch={h+1}.npz"
            np.savez(file = os.path.join(dirname, 'Results', filename), reg_param = lambda_param_value, weights = weight_est, loss = loss_est)



    
print(f"Optimization finished")



# save results
filename = f"1d-spline_{x1_name}_method={method}_basisdim={basis_dimension}_epochs={num_epochs_splines}.npz"
np.savez(file = os.path.join(dirname, 'Results', filename), reg_param = lambda_param_value, weights = weight_est)
