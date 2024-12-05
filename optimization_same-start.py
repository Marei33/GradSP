import pandas as pd
import tensorflow as tf
import numpy as np
import os as os
import matplotlib.pyplot as plt

# own code base
import tf_loss_functions as lf
import splines as sp


#####################
# Parameter Setting #
#####################


# set working directory
os.chdir(os.path.dirname(__file__))
dirname = os.path.dirname(__file__)

# set spline hyperparameters
basis_dimension = 20  # number of B-spline basis functions
degree_bsplines = 3  # degree of B-splines
penalty_diff_order = 2  # order of difference penalty


# set optimization parameters
weights_init_mean = 0.0
weights_init_sd = 0.1
num_epochs_lambda_param = 10000
num_epochs_splines = 1
learning_rate_lambda_param = 1
learning_rate_splines = 1

methods = ["gcv", "reml"]  # reml or gcv

k_range = ["linear", "wiggly", "random"]


#####################
# Optimization Loop #
#####################


# loop over both methods
for method in methods:

    lambda_param_est_opt = np.zeros([3, 10])

    # loop over every data generation type
    for k in range(len(k_range)):
        print(f"Next Data Generation Types {k}")

        data_x = pd.read_csv(f"./data/df_x_{k_range[k]}.csv", index_col=0)
        data_y = pd.read_csv(f"./data/df_y_{k_range[k]}.csv", index_col=0)

        # loop over every data set per data generation type
        for j in range(0, 10):

            print(f"Data set {j}")

            lambda_param_est_values = []
            pen_loss_values = []
            epochs_saved = []

            ### data preparation ###

            x1 = data_x.iloc[j]
            x1_name = f"{k_range[k]}_{j}"

            # set up pspline
            x1_ps = sp.pspline(
                x=x1,
                degree_bsplines=degree_bsplines,
                penalty_diff_order=penalty_diff_order,
                knot_type="equi",
                basis_dimension=basis_dimension,
            )

            labels = data_y.iloc[j]
            labels = np.float32((labels - labels.mean()) / labels.std())
            labels = np.expand_dims(labels, 1)

            ### model building and optimization ###

            # same initial starting point for penalty coefficient instead of grid search

            lambda_param_init = 10e0
            if method == "gcv":
                value = lf.gcv_1d(
                    y=labels,
                    design_matrix_Z=np.float32(x1_ps.design_matrix),
                    reg_matrix_K=np.float32(x1_ps.penalty_matrix),
                    reg_param=tf.exp(lambda_param_init),
                )
            if method == "reml":
                value = lf.reml_1d(
                    y=labels,
                    design_matrix_Z=np.float32(x1_ps.design_matrix),
                    reg_matrix_K=np.float32(x1_ps.penalty_matrix),
                    reg_param=tf.exp(lambda_param_init),
                )

            # save start values
            epochs_saved.append(0)
            lambda_param_est_values.append(lambda_param_init)
            pen_loss_values.append(value.numpy().item())

            # set parameters for optimization
            initializer = tf.keras.initializers.TruncatedNormal(
                mean=weights_init_mean, stddev=weights_init_sd, seed=13
            )
            opt_lambda_param = tf.keras.optimizers.Adam(
                learning_rate=learning_rate_lambda_param
            )
            opt_splines = tf.keras.optimizers.Adam(learning_rate=learning_rate_splines)

            lambda_param = tf.Variable(
                lambda_param_init, name="lambda_param", dtype=tf.float32, trainable=True
            )
            weights = tf.Variable(
                initializer(shape=(basis_dimension, 1)), name="weights"
            )

            # Loop for optimization
            # 1) update smoothing parameter lambda
            for i in range(num_epochs_lambda_param):
                if method == "gcv":
                    loss = lambda: lf.gcv_1d(
                        y=labels,
                        design_matrix_Z=x1_ps.design_matrix_d,
                        reg_matrix_K=x1_ps.penalty_matrix_d,
                        reg_param=tf.exp(lambda_param),
                    )
                if method == "reml":
                    loss = lambda: lf.reml_1d(
                        y=labels,
                        design_matrix_Z=x1_ps.design_matrix_d,
                        reg_matrix_K=x1_ps.penalty_matrix_d,
                        reg_param=tf.exp(lambda_param),
                    )
                opt_lambda_param.minimize(loss, var_list=[lambda_param])
                pen_loss = loss().numpy().item()

                # 2) update spline coefficients
                for h in range(num_epochs_splines):
                    loss = lambda: lf.penalized_least_squares(
                        y=labels,
                        weights=weights,
                        design_matrix=x1_ps.design_matrix_d,
                        reg_param=tf.exp(lambda_param),
                        penalty_matrix=x1_ps.penalty_matrix_d,
                    )
                    opt_splines.minimize(loss, var_list=[weights])

                # save every 100th epoch
                if (i + 1) % 100 == 0:
                    print(f"Epoch: {i+1}")
                    epochs_saved.append(i + 1)
                    lambda_param_est = tf.exp(lambda_param).numpy()
                    lambda_param_est_values.append(lambda_param_est)
                    pen_loss_values.append(pen_loss)

            # save lambda param and according loss every 100 epoch
            training_frame = pd.DataFrame(
                data=np.array(
                    [epochs_saved, lambda_param_est_values, pen_loss_values]
                ).T,
                columns=["Epoch", "Lambda Parameter", f"{method}"],
            )
            training_frame.to_csv(
                f"Results/Smoothing_param_method={method}_epochs={num_epochs_lambda_param}_{k}_j{j}.csv"
            )

            weight_est = np.dot(x1_ps.U, weights.numpy())
            lambda_param_est = tf.exp(lambda_param).numpy()
            lambda_param_est_opt[k, j] = lambda_param_est
            print(f"Optimization finished \nSmoothing parameter: {lambda_param_est}")

            # save final results
            filename = (
                "1d-spline_"
                + x1_name
                + "_method="
                + method
                + "_basisdim="
                + str(basis_dimension)
                + "_epochs="
                + str(num_epochs_lambda_param * num_epochs_splines)
                + ".npz"
            )
            np.savez(
                file=os.path.join(dirname, "Results", filename),
                reg_param=lambda_param_est,
                weights=weight_est,
            )

    # save optimal lambda value for every data set per method
    lambda_param_est_opt = pd.DataFrame(data=lambda_param_est_opt)
    lambda_param_est_opt.to_csv(
        f"Results/Smoothing_param_OPT_method={method}_epochs={num_epochs_lambda_param}.csv"
    )
