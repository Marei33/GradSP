
import numpy as np


def single_b_spline (x, knots, degree, index):
    res = np.repeat(0, len(x))
## building indicator function
    if(degree < 0.5):
        for i in range(len(x)):
            res[i] = 1  * (x[i] >= knots[index] and x[i] < knots[index+1])      # TRUE = 1, FALSE = 0  
## Recursive call to single.b.spline from inside the same function:
    else:
        res = (x-knots[index]) / (knots[index+degree]-knots[index]) * single_b_spline(x, knots, degree-1, index)
        res = res + (knots[index+degree+1]-x)/(knots[index+degree+1]-knots[index+1])*single_b_spline(x, knots, degree-1, index+1)
    return(res)


def build_base (x, degree, knot_type="equi", basis_dimension=None, given_knots=None):
    '''
    x: vector of the values of the predictor variable
    Returns a list where first element (base) is the design matrix and the
    second element (knots) is a vector of the knots.    
    '''
## input check
    if (knot_type != "equi" and knot_type != "given"):
        raise ValueError('knot_type has to be specified as either "equi" or "given".')
    if (knot_type == "equi" and basis_dimension == None):
        raise ValueError('For knot_type = "equi" the basis dimension has to be specified.')
    if (knot_type == "given" and any(given_knots == None)):
        raise ValueError('knot_type = "given" was chosen, but no knots were supplied.')
## see Wood, Generalized Additive Models, p.204, for construction of p-splines
## building the knots:
    if (knot_type == "given"):
        knots = given_knots
        basis_dimension = len(given_knots) - degree - 1
    if (knot_type == "equi"):
        n_inner_knots = basis_dimension - degree + 1
        inner_knots = np.linspace(start = min(x), stop = max(x), num = n_inner_knots)     # min(x) & max(x) sind knots
        min_dist = min(np.diff(inner_knots))
        left_outer_knots = np.linspace(start = min(x)-degree*min_dist, stop = min(x)-min_dist, num = degree)
        right_outer_knots = np.linspace(start = max(x)+min_dist, stop = max(x)+degree*min_dist, num = degree)
        knots = np.append(arr=left_outer_knots, values = inner_knots)
        knots = np.append(arr=knots, values = right_outer_knots)
## Building the base:   
    index = np.arange(start = 0, stop = basis_dimension, step = 1)
    base = single_b_spline(x, knots, degree, index[0])
    for i in index[1:]: 
        base = np.c_[base, single_b_spline(x, knots, degree, i)]
    return list([base, knots])


def build_penalty_matrix(diff_order, dim):
    '''
    dim is the number of columns of the designmatrix (basis dimension)
    '''
    I = np.identity(dim, dtype = 'float32')
    D_t = np.diff(I, n = diff_order)
    K = np.matmul(D_t, D_t.T)
    return K


def diagonalize_penalty_matrix(penalty_matrix):
    ''' 
    Returns diagonalized regularization matrix and orthogonal matrix of
    eigenvectors U. Note: reg_matrix = U * reg_matrix_diag * U.T 
    '''
    eigval, U = np.linalg.eigh(penalty_matrix)
    penalty_matrix_diag = np.diag(eigval)
    return (penalty_matrix_diag, U)


class pspline():
    """
    Creates design_ matrix and penalty matrix for a p-spline.
    """

    def __init__(self, x, degree_bsplines, penalty_diff_order, knot_type="equi", basis_dimension=None, given_knots=None):
        # input check
        if (knot_type != "equi" and knot_type != "given"):
            raise ValueError('knot_type has to be specified as either "equi" or "given".')
        if (knot_type == "equi" and basis_dimension == None):
            raise ValueError('For knot_type = "equi" the basis dimension has to be specified.')
        if (knot_type == "given" and any(given_knots == None)):
            raise ValueError('knot_type = "given" was chosen, but no knots were supplied.')

        self.degree_bsplines = degree_bsplines        
        self.penalty_diff_order = penalty_diff_order
     
        if (knot_type == "equi"):
            self.basis_dimension = basis_dimension
            self.design_matrix, self.knots = build_base(x, degree=self.degree_bsplines, basis_dimension = self.basis_dimension)
        if (knot_type == "given"):
            self.basis_dimension = len(given_knots) - self.degree_bsplines - 1
            self.design_matrix, self.knots = build_base(x, degree=self.degree_bsplines, knot_type="given", given_knots=given_knots)

        self.penalty_matrix = build_penalty_matrix(self.penalty_diff_order, dim=self.basis_dimension)
        self.penalty_matrix_d, self.U = diagonalize_penalty_matrix(self.penalty_matrix)
        self.design_matrix_d = np.float32(np.matmul(self.design_matrix, self.U))
               