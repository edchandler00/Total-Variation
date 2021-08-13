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











# # b = None # observed image
# # reg = None # regulariztion param
# # N = None # num iter

# def fgp(b, reg, N):

#     m,n = b.shape[:2]

#     p_array = []
#     q_array = []

#     p_array.append(np.zeros((m-1, n)))
#     q_array.append(np.zeros((m, n-1)))

#     r_array = []
#     s_array = []

#     # Setting this for consitency in notation
#     r_array.append(np.empty((m-1,n)))
#     s_array.append(np.empty((m,n-1)))

#     r_array.append(np.zeros((m-1, n)))
#     s_array.append(np.zeros((m, n-1)))

#     t_array = []
#     t_array.append(np.nan) # Setting this for consitency in notation
#     t_array.append(1)

#     for k in tqdm(range(1, N+1)):
#         # Following matlab notation
#         D = util.proj_C(b - reg*util.L(r_array[k], s_array[k]))
#         Q = util.L_t(D)
#         p_k = util.proj_P( (r_array[k]) + 1/(8*reg) * Q[0] , 'iso')
#         q_k = util.proj_P( (s_array[k]) + 1/(8*reg) * Q[1] , 'iso')
#         # p_k, q_k = util.proj_P( (r_array[k], s_array[k]) + 1/(8*reg) * Q , 'iso')
#         p_array.append(p_k)
#         q_array.append(q_k)

#         t_array.append((1 + np.sqrt(1+4* (t_array[k]**2) )) / 2)

#         # r_kplus1, s_kplus1 = (p_array[k], q_array[k]) + ((t_array[k] - 1)/t_array[k+1]) (p_array[k]-p_array[k-1], q_array[k]-q_array[k-1] )
#         r_kplus1 = p_array[k] + ((t_array[k] - 1)/t_array[k+1]) * (p_array[k]-p_array[k-1])
#         s_kplus1 = q_array[k] + ((t_array[k] - 1)/t_array[k+1]) * (q_array[k]-q_array[k-1])

#         r_array.append(r_kplus1)
#         s_array.append(s_kplus1)

#         # TODO: implement re 

#     x_star = util.proj_C(b - reg*util.L(p_array[N], q_array[N]))

#     print(f'Found Optimal')
#     return x_star

#     # plt.imshow(x_star)
#     # plt.show()