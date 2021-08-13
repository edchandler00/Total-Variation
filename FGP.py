import numpy as np
import util
from tqdm import tqdm
import matplotlib.pyplot as plt

def fgp(b, reg, l, u, param_init, max_iter, epsilon=1e-4, tv_type='iso', print_info=True):
    """
    b: Observed noisy image,
    reg: regularization parameter lambda,
    l: Lower bound constraint,
    u: Upper bound constraint,
    param_init: how to initialize p, q, r, and s,
    max_iter: max number of iterations,
    epsilon: Stopping criteria,
    tv_type: The choice of Total Varation (iso or l1)
    print_info: Whether or not to print the optimization information at each step (True of False)
    """
    m,n = b.shape[:2]

    if param_init:
        p, q = param_init[0], param_init[1]
        r, s = param_init[0], param_init[1]
    else:
        p, q = np.zeros((m-1,n)), np.zeros((m,n-1))
        r, s = np.zeros((m-1,n)), np.zeros((m,n-1))

    D = np.zeros((m,n))
    t_kplus1 = 1
    # fval = np.inf
    threshold_count = 0 # Keep track of the number of times below threshold

    for k in range(1, max_iter+1):
        # fold = fval
        D_old = D
        p_old = p
        q_old = q
        t_k = t_kplus1

        # Following matlab notation
        D = util.proj_C(b - reg*util.L(r, s), l, u)
        Q = util.L_t(D)
        p,q = util.proj_P( r + 1/(8*reg) * Q[0], s + 1/(8*reg) * Q[1] , tv_type)

        t_kplus1 = (1 + np.sqrt(1+4* (t_k**2) )) / 2

        r = p + ((t_k - 1)/t_kplus1) * (p-p_old)
        s = q + ((t_k - 1)/t_kplus1) * (q-q_old)


        re = np.linalg.norm(D-D_old, 'fro')/np.linalg.norm(D,'fro')
        if re < epsilon:
            threshold_count += 1
        else:
            threshold_count = 0

        # TODO: figure out what this is
        C = b - reg*util.L(p,q)
        PC = util.proj_C(C, l,u)
        fval = -((np.linalg.norm(C-PC, ord='fro'))**2) + (np.linalg.norm(C, 'fro'))**2
        if print_info:
            print(f'iter {k}\t\t{fval}\t\t{re}')
    
        if threshold_count >= 5:
            break


    # x_star = util.proj_C(b - reg*util.L(p_array[max_iter], q_array[max_iter]))
    paper_x_star = util.proj_C(b - reg*util.L(p, q), l, u) # The final value as per the algorithm on the paper
    code_x_star = D # The final value as per the matlab code

    x_star = code_x_star # FIXME: The disparity between the paper and the code?!?! which is best/correct?
    return x_star, k, (p,q)