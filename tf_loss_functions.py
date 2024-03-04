import tensorflow as tf
import numpy as np


def penalized_least_squares(y, weights, design_matrix, reg_param, penalty_matrix):
    y_hat = tf.matmul(design_matrix, weights)
    sq_diff = tf.reduce_sum(input_tensor=(y - y_hat) ** 2)
    penalty = reg_param * (
        tf.tensordot(
            tf.transpose(weights), tf.tensordot(penalty_matrix, weights, axes=1), axes=1
        )
    )
    loss = sq_diff + penalty
    return loss


def gcv_2d(
    y,
    design_matrix_Z_1,
    reg_matrix_K_1,
    reg_param_1,
    design_matrix_Z_2,
    reg_matrix_K_2,
    reg_param_2,
):

    design_matrix_Z = np.float32(
        tf.concat([design_matrix_Z_1, design_matrix_Z_2], axis=1)
    )

    G1 = (
        tf.linalg.matmul(a=design_matrix_Z_1, b=design_matrix_Z_1, transpose_a=True)
        + reg_param_1 * reg_matrix_K_1
    )
    G2 = tf.linalg.matmul(a=design_matrix_Z_1, b=design_matrix_Z_2, transpose_a=True)
    G3 = tf.linalg.matmul(a=design_matrix_Z_2, b=design_matrix_Z_1, transpose_a=True)
    G4 = (
        tf.linalg.matmul(a=design_matrix_Z_2, b=design_matrix_Z_2, transpose_a=True)
        + reg_param_2 * reg_matrix_K_2
    )

    G = tf.concat([tf.concat([G1, G2], axis=1), tf.concat([G3, G4], axis=1)], axis=0)

    G_inv = tf.linalg.inv(G)
    S = design_matrix_Z @ G_inv @ tf.linalg.matrix_transpose(design_matrix_Z)

    n = y.shape[0]
    alpha_0 = y - tf.linalg.matmul(S, y)
    alpha = tf.linalg.matmul(a=alpha_0, b=alpha_0, transpose_a=True)
    delta = n - tf.linalg.trace(S)

    gcv = n * alpha / delta**2
    print(f"gcv: {gcv}")
    return gcv


def gcv_1d(y, design_matrix_Z, reg_matrix_K, reg_param):

    G = (
        tf.linalg.matrix_transpose(design_matrix_Z) @ design_matrix_Z
        + reg_param * reg_matrix_K
    )
    G_inv = tf.linalg.inv(G)
    S = design_matrix_Z @ G_inv @ tf.linalg.matrix_transpose(design_matrix_Z)

    n = y.shape[0]
    alpha_0 = y - tf.linalg.matmul(S, y)
    alpha = tf.linalg.matmul(a=alpha_0, b=alpha_0, transpose_a=True)
    delta = n - tf.linalg.trace(S)

    gcv = n * alpha / delta**2
    return gcv


def reml_1d(y, design_matrix_Z, reg_matrix_K, reg_param):
    n = y.shape[0]
    G = (
        tf.linalg.matrix_transpose(design_matrix_Z) @ design_matrix_Z
        + reg_param * reg_matrix_K
    )
    S_0 = tf.linalg.matmul(a=tf.linalg.inv(G), b=design_matrix_Z, transpose_b=True)
    S = design_matrix_Z @ S_0

    alpha_0 = y - tf.linalg.matmul(S, y)
    alpha = tf.linalg.matmul(a=alpha_0, b=alpha_0, transpose_a=True)
    beta_est = tf.linalg.matmul(S_0, y)
    phi = alpha / (n - tf.linalg.trace(S_0 @ design_matrix_Z))

    reml1 = (
        (
            alpha
            + tf.linalg.matmul(
                a=beta_est, b=(reg_param * reg_matrix_K), transpose_a=True
            )
            @ beta_est
        )
        / 2
        * phi
    )

    reml2 = (
        tf.math.log(tf.linalg.det(G / tf.norm(G, ord="euclidean")))
        + G.shape[0] * tf.math.log(tf.norm(G, ord="euclidean"))
        - G.shape[0] * tf.math.log(phi)
    ) / 2

    
    reml3 = (
        -tf.cast(np.linalg.matrix_rank(reg_matrix_K), dtype=tf.float32)
        * tf.math.log(reg_param / phi)
        / 2
    )

    reml = reml1 + reml2 + reml3
    return reml
