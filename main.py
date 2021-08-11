import scipy.io
import numpy as np
from FGP import fgp
import matplotlib.pyplot as plt
import util
import MFISTA


# Bobs = scipy.io.loadmat('cameraman_Bobs.mat')['Bobs']
# print(Bobs.dtype)
# print(f'Bobs {Bobs}')
# # plt.imshow(Bobs, cmap="gray", vmin=0, vmax=1)
# x_star, _, _ = fgp(b=Bobs, reg=0.02, l=-np.inf, u=np.inf, P_init=[], N=100, epsilon=1e-4, tv_type="iso", print_info=True)

# plt.imshow(x_star, cmap="gray", vmin=0, vmax=1)
# plt.show()




# a = np.array([[.1, .3, .55], [.22, .78, .09]])
# print(util.total_var(a, 'iso'))
# print(util.total_var(a, 'l1'))



Bobs = scipy.io.loadmat('cameraman_Bobs_blurry.mat')["Bobs"]
PSF = 1/9 * np.ones((3,3))
P_center = np.array([1,1])


subprob_params = {
    'max_iter': 10, 
    'epsilon': 1e-5,
    'tv_type': 'iso'
}
X_deblur = MFISTA.mfista(Bobs=Bobs, PSF=PSF, P_center=P_center, reg=0.001, l=-np.inf, u=np.inf, max_iter=100, BC='reflexive', tv_type='iso', subprob=subprob_params, show_fig=True)

# subprob_params = {
#     'max_iter': 10, 
#     'epsilon': 1e-5,
#     'tv_type': 'iso'
# }
# X_deblur = MFISTA.mfista(Bobs=Bobs, PSF=PSF, P_center=P_center, reg=0.001, l=-np.inf, u=np.inf, max_iter=20, BC='periodic', tv_type='iso', subprob=subprob_params)

plt.imshow(X_deblur, cmap="gray", vmin=0, vmax=1)
plt.show()