import numpy as np
from FGP import fgp
import util
import scipy.fftpack
import matplotlib.pyplot as plt

def mfista(Bobs, PSF, P_center, reg, l, u, max_iter, BC, tv_type, subprob, show_fig=False):
    # TODO: check about P_center indicing. currently using 0 
    m, n = Bobs.shape[:2]

    # Padding PSF
    Pbig = util.padPSF(PSF, m, n)

    if BC == "reflexive":
        def trans(the_input):
            return scipy.fftpack.dctn(the_input, type=2, norm="ortho")
        def inv_trans(the_input):
            return scipy.fftpack.dctn(the_input, type=3, norm="ortho") 
        e1 = np.zeros((m,n))
        e1[0,0] = 1 # what is this for??
        Sbig = scipy.fftpack.dctn(util.dctshift(Pbig, P_center) , type=2, norm="ortho") / scipy.fftpack.dctn(e1 , type=2, norm="ortho")
    elif BC == "periodic":  #FIXME: this is not working!! 
        def trans(the_input):
            return 1/np.sqrt(m*n) * np.fft.fft2(the_input)
        def inv_trans(the_input):
            return np.sqrt(m*n) * np.fft.ifft2(the_input)

        # TODO: make sure I don't need 1-P_center (I don't think I do because of diff in indexing)
        # TODO: figure out why -P_center and the compute of eigenvalues of blurring matrix
        Sbig = np.fft.fft2( util.circshift(Pbig, -P_center) ) # this is good when using 0 indexing
    else:
        quit("ruh roh")

    Btrans = trans(Bobs) # TODO: figure out what this is!!!!

    #FIXME: change back!!!!!!!
    L = 2 * np.max(  (np.abs(Sbig))**2 , axis=None)  # Lipschitz constant TODO: figure out what this is

    x_k = Bobs
    y_kplus1 = x_k
    t_kplus1 = 1
    
    func_vals = []

    for k in range(1, max_iter+1):
        x_old = x_k
        t_k = t_kplus1
        y_k = y_kplus1

        D = Sbig * trans(y_k) - Btrans

        y_k = y_k - 2/L * inv_trans(np.conj(Sbig) * D) 
        y_k = np.real(y_k) # TODO: figure out why might be complex???

        if k == 1:
            z_k, num_iter, (p,q) = fgp(y_k, (2*reg)/L, l, u, [], subprob['max_iter'], subprob['epsilon'], subprob['tv_type'], False)
        else:
            z_k, num_iter, (p,q) = fgp(y_k, (2*reg)/L, l, u, [p,q], subprob['max_iter'], subprob['epsilon'], subprob['tv_type'], False)


        # This is MFISTA
        total_var = util.total_var(z_k, tv_type)
        func_val = np.linalg.norm(Sbig * trans(z_k) - Btrans, 'fro')**2 + 2*reg*total_var # TODO: figure this out!!

        if (k > 1):
            if func_val > func_vals[-1]:
                x_k = x_old
                func_val = func_vals[-1]
            else:
                x_k = z_k

        func_vals.append(func_val) # TODO: make sure this is okay (they only do when nargs is 2, but I think that's wrong)
        t_kplus1 = (1+np.sqrt(1 + 4 * (t_k**2) )) / 2
        y_kplus1 = x_k + (t_k / t_kplus1) * (z_k - x_k) + (t_k - 1) / t_kplus1 * (x_k - x_old)

        print(f'{k}\t\t{func_val:.5f}\t\t{total_var:.5f}\t\t{num_iter}\t\t{np.linalg.norm(x_k-x_old, "fro")/np.linalg.norm(x_old,"fro"):.5f}')

        if show_fig:
            if k == 1:
                the_pic = plt.imshow(x_k, cmap="gray", vmin=0, vmax=1)
            else:
                the_pic.set_data(x_k)
            plt.draw()
            plt.pause(0.0001)

    # plt.show()

    x_star = x_k
    return x_star