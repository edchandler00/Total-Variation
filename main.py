import scipy.io
import numpy as np
import fgp
import matplotlib.pyplot as plt
import util
import MFISTA
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--BoundaryCondition", type=int, default=1, help="1: optimize cameraman_Bobs_blurry with reflexive boundary condition; 2: optimize cameraman_Bobs_blurry with periodic boundary condition. Default is 1.")
    parser.add_argument("-s", "--ShowFig", type=int, default=0, help="Whether or not show image as optimizing. Default is 0.")
    
    args = parser.parse_args()
    opt_prob = args.BoundaryCondition
    show_fig = bool(args.ShowFig)

    the_dict = scipy.io.loadmat('cameraman_Bobs_blurry.mat')
    Bobs = the_dict["Bobs"]
    PSF = the_dict["P"]
    P_center = the_dict["center"][0] - [1,1] # must subtract by one to switch from matlab 1-index to python 0-index

    if opt_prob == 1:
        subprob_params = {
            'max_iter': 10, 
            'epsilon': 1e-5,
        }
        X_deblur = MFISTA.mfista(b=Bobs, P=PSF, P_center=P_center, reg=0.001, l=-np.inf, u=np.inf, max_iter=100, boundary_condition='reflexive', tv_type='iso', subprob=subprob_params, show_fig=show_fig)
    elif opt_prob == 2:
        subprob_params = {
            'max_iter': 10, 
            'epsilon': 1e-5,
        }
        X_deblur = MFISTA.mfista(b=Bobs, P=PSF, P_center=P_center, reg=0.001, l=-np.inf, u=np.inf, max_iter=20, boundary_condition='periodic', tv_type='iso', subprob=subprob_params, show_fig=show_fig)
    else:
        quit("Not Valid")

    fig, (ax1, ax2) = plt.subplots(1,2)

    ax1.imshow(Bobs, cmap="gray", vmin=0, vmax=1)
    ax2.imshow(X_deblur, cmap="gray", vmin=0, vmax=1)
    plt.show()


# # This commented out is to denoise only with FGP
# Bobs = scipy.io.loadmat('cameraman_Bobs.mat')['Bobs']
# print(Bobs.dtype)
# print(f'Bobs {Bobs}')
# # plt.imshow(Bobs, cmap="gray", vmin=0, vmax=1)
# x_star, _, _ = FGP.fgp(b=Bobs, reg=0.02, l=-np.inf, u=np.inf, param_init=[], max_iter=100, epsilon=1e-4, tv_type="iso", print_info=True)

# plt.imshow(x_star, cmap="gray", vmin=0, vmax=1)
# plt.show()
