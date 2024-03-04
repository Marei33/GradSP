# %%
import pandas as pd
import tensorflow as tf
import numpy as np
import os
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

k_range = ["linear", "wiggly", "random"]


# %%

####################
# Calculation Loop #
####################


# loop over every data generation type
for k in range(len(k_range)):

    data_x = pd.read_csv(f"./df_x_{k_range[k]}.csv", index_col=0)
    data_y = pd.read_csv(f"./df_y_{k_range[k]}.csv", index_col=0)

    lambda_size = 100
    lambda_param = np.logspace(-2, 6, lambda_size, dtype=np.float32)

    gcv_values = np.zeros([lambda_size, 10])
    reml_values = np.zeros([lambda_size, 10])

    # loop over every data set per data generation type
    for j in range(10):

        x1 = data_x.iloc[j]

        x1_ps = sp.pspline(
            x=x1,
            degree_bsplines=degree_bsplines,
            penalty_diff_order=penalty_diff_order,
            knot_type="equi",
            basis_dimension=basis_dimension,
        )

        labels = data_y.iloc[j]
        labels = np.float32((labels - labels.mean()) / labels.std())
        labels_expanded = np.expand_dims(labels, 1)

        for i in range(lambda_size):
            gcv_values[i, j] = lf.gcv_1d(
                y=labels_expanded,
                design_matrix_Z=np.float32(x1_ps.design_matrix),
                reg_matrix_K=np.float32(x1_ps.penalty_matrix),
                reg_param=lambda_param[i],
            )

            reml_values[i, j] = lf.reml_1d(
                y=labels_expanded,
                design_matrix_Z=np.float32(x1_ps.design_matrix),
                reg_matrix_K=np.float32(x1_ps.penalty_matrix),
                reg_param=lambda_param[i],
            )

    gcv_values = pd.DataFrame(data=gcv_values)
    reml_values = pd.DataFrame(data=reml_values)

    gcv_values.to_csv(f"./GCV_values_{k_range[k]}.csv")
    reml_values.to_csv(f"./REML_values_{k_range[k]}.csv")
