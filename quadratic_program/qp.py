import numpy as np
import quadprog

def quadprog_wrapper(W, X, l, n, t):
    """
    :param W: is a vector of weight vectors [w_1, ..., w_l] or list of lists
    :param X: is a vector of Pareto points [r_1, ..., r_l] or list of lists
    :param l: length of the current hullset
    :param n: the number of objectives n + m
    """
    # construct a vector of component wise w.r
    print(W)
    h = np.array([np.dot(np.array(W[i]), np.array(X[i])) for i in range(0, l)])
    P = np.eye(n)
    q = -np.dot(np.transpose(np.eye(n)), np.array(t))
    return quadprog_solve_qp(P, q, np.array(W), h)


def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]