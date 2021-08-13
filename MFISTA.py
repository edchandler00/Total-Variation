import numpy as np
from FGP import fgp
import util
import scipy.fftpack
import matplotlib.pyplot as plt

def mfista(b, P, P_center, reg, l, u, max_iter, boundary_condition, tv_type, subprob, show_fig=False):
    """
    b: observed blurry/noisy image
    P: PSF
    P_center: Center coordinates of PSF (Note: should be 0-indexed, unlike the MATLAB code)
    reg: The regularization parameter lambda
    l: Lower bound constraint
    u: Upper bound constraint
    max_iter: Maximum number of iterations for MFISTA to run
    boundary_condition: The choice of boundary condition (reflexive or periodic)
    tv_type: The choice of Total Varation (iso or l1)
    subprob["max_iter"]: Maximum number of iterations to run the denoising subproblem on each iteration of MFISTA
    subprob["epsilon"]: Denoising subproblem stopping criteria
    show_fig: Whether or not to show the image as optimize (runs slower) (True or False)
    """
    m, n = b.shape[:2]

    Pbig = util.padPSF(P, m, n)
    if boundary_condition == "reflexive":
        def trans(the_input):
            return scipy.fftpack.dctn(the_input, type=2, norm="ortho")
        def inv_trans(the_input):
            return scipy.fftpack.dctn(the_input, type=3, norm="ortho") 
        temp = np.zeros((m,n))
        temp[0,0] = 1
        # TODO: Is this the exact same as A? Or is it slightly different (the code doesn't follow the equations exactly 
        # if it is meant to be A... I think).
        Sbig = scipy.fftpack.dctn(util.dctshift(Pbig, P_center) , type=2, norm="ortho") / scipy.fftpack.dctn(temp , type=2, norm="ortho")
    elif boundary_condition == "periodic": 
        def trans(the_input):
            return 1/np.sqrt(m*n) * np.fft.fft2(the_input)
        def inv_trans(the_input):
            return np.sqrt(m*n) * np.fft.ifft2(the_input)

        Sbig = np.fft.fft2( util.circshift(Pbig, -P_center) )
    else:
        quit("Invalid boundary condition (var: boundary_condition). Must be either 'reflexive' or 'periodic'") # TODO: Throw an error

    b_trans = trans(b) # TODO: figure out why need to do this

    L = 2 * np.max(  (np.abs(Sbig))**2 , axis=None)  # Lipschitz constant TODO: learn what this is

    x_k = b
    y_kplus1 = x_k
    t_kplus1 = 1
    
    func_vals = []

    if show_fig:
        the_pic = plt.imshow(x_k, cmap="gray", vmin=0, vmax=1)

    for k in range(1, max_iter+1):
        x_old = x_k
        t_k = t_kplus1
        y_k = y_kplus1

        y_k_error = Sbig * trans(y_k) - b_trans # equiv to variable D in matlab code. I think this variable name makes sense for what it is.

        y_k = y_k - 2/L * inv_trans(np.conj(Sbig) * y_k_error) 
        y_k = np.real(y_k) # TODO: figure out why might be complex???

        if k == 1:
            z_k, num_iter, (p,q) = fgp(b=y_k, reg=(2*reg)/L, l=l, u=u, param_init=[], max_iter=subprob['max_iter'], epsilon=subprob['epsilon'], tv_type=tv_type, print_info=False)
        else:
            z_k, num_iter, (p,q) = fgp(b=y_k, reg=(2*reg)/L, l=l, u=u, param_init=[p,q], max_iter=subprob['max_iter'], epsilon=subprob['epsilon'], tv_type=tv_type, print_info=False)


        # Only implementing MFISTA (and not also FISTA), so I don't have the outer conditional that is in matlab code
        total_var = util.total_var(z_k, tv_type)
        func_val = np.linalg.norm(Sbig * trans(z_k) - b_trans, 'fro')**2 + 2*reg*total_var # This is just F(x): f(x) is L side of + and g(x) is R side

        if (k > 1):
            if func_val > func_vals[-1]:
                x_k = x_old
                func_val = func_vals[-1]
            else:
                x_k = z_k

        func_vals.append(func_val)
        
        t_kplus1 = (1+np.sqrt(1 + 4 * (t_k**2) )) / 2
        y_kplus1 = x_k + (t_k / t_kplus1) * (z_k - x_k) + (t_k - 1) / t_kplus1 * (x_k - x_old)

        print(f'{k}\t\t{func_val:.5f}\t\t{total_var:.5f}\t\t{num_iter}\t\t{np.linalg.norm(x_k-x_old, "fro")/np.linalg.norm(x_old,"fro"):.5f}')

        if show_fig:
            the_pic.set_data(x_k)
            plt.draw()
            plt.pause(0.0001)


    x_star = x_k
    return x_star