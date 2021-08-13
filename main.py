import scipy.io
import numpy as np
import FGP
import matplotlib.pyplot as plt
import util
import MFISTA
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--Input", default="cameraman_Bobs_blurry.mat", help="Input mat file. Default is cameraman_Bobs_blurry.mat")
    parser.add_argument("-p", "--OptimizationTask", default="deblur", help="Whether to deblur/denoise or denoise only. Default is deblur")
    parser.add_argument("-t", "--TVType", default="iso", help="Total variation type (iso or l1). Default is iso")
    parser.add_argument("-n", "--MaxIter", type=int, default=100, help="Max number of iterations to run")
    parser.add_argument("-l", "--LowerBound", type=float, default=-np.inf, help="Lower bound constraint. Default is -Inf")
    parser.add_argument("-u", "--UpperBound", type=float, default=np.inf, help="Lower bound constraint. Default is Inf")
    parser.add_argument("-b", "--BoundaryCondition", default="reflexive", help="reflexive boundary condition or periodic boundary condition. Default is reflexive")
    parser.add_argument("-s", "--ShowFig", type=int, default=0, help="Whether or not show image as optimizing. Default is 0")
    
    args = parser.parse_args()

    input_file = args.Input

    opt_task = args.OptimizationTask
    tv_type = args.TVType
    max_iter = args.MaxIter
    l = args.LowerBound
    u = args.UpperBound
    boundary_condition = args.BoundaryCondition
    show_fig = bool(args.ShowFig)

    the_dict = scipy.io.loadmat(input_file)
    Bobs = the_dict["Bobs"]
    
    if opt_task == "deblur":
        PSF = the_dict["P"] if "P" in the_dict.keys() else np.array([[0.,0.,0.], [0.,1.,0.], [0.,0.,0.]])
        P_center = the_dict["center"][0] - [1,1] if "center" in the_dict.keys() else [1,1] # must subtract by one to switch from matlab 1-index to python 0-index
        subprob_params = {
            'max_iter': 10, 
            'epsilon': 1e-5,
        }
        X_recon = MFISTA.mfista(b=Bobs, P=PSF, P_center=P_center, reg=.001, l=l, u=u, max_iter=max_iter, boundary_condition=boundary_condition, tv_type=tv_type, subprob=subprob_params, show_fig=show_fig)
    elif opt_task == "denoise":
        X_recon, _, _ = FGP.fgp(b=Bobs, reg=0.02, l=l, u=u, param_init=[], max_iter=max_iter, epsilon=1e-4, tv_type=tv_type, print_info=True, show_fig=show_fig)
    else:
        quit("Not Valid")

    fig, (ax1, ax2) = plt.subplots(1,2)

    ax1.imshow(Bobs, cmap="gray", vmin=0, vmax=1)
    ax1.set_title("Blurry/Noisy")
    ax2.imshow(X_recon, cmap="gray", vmin=0, vmax=1)
    ax2.set_title("Reconstructed")
    plt.show()