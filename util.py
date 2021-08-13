import numpy as np

def total_var(x, tv_type):
    m, n = x.shape[:2]
    P = L_t(x) # TODO: wait, what?? why do this?
    to_return = 0

    if tv_type == "iso":
        D = np.zeros((m,n))
        D[:m-1,:] = P[0]**2
        D[:,:n-1] = D[:,:n-1] + P[1]**2
        return np.sum(np.sqrt(D))
        
    elif tv_type == "l1":
        return np.sum(np.abs(P[0])) + np.sum(np.abs(P[1]))
    else:
        quit("Invalid Total Variation type (var: tv_type). Must be either 'iso' or 'l1'")
    
    return to_return

def L(p,q):
    m = q.shape[0]
    n = p.shape[1]

    p_padded = np.pad(p, ((1,1), (0,0)), 'constant', constant_values=0) # pad top and bottom
    q_padded = np.pad(q, ((0,0), (1,1)), 'constant', constant_values=0) # pad left and right

    L_output = p_padded[1:m+1,:n] + q_padded[:m,1:n+1] - p_padded[:m,:n] - q_padded[:m,:n]

    return L_output
    
def L_t(x): # x is mxn
    m, n = x.shape[:2]
    p = np.empty((m-1, n)) # (m-1, n)
    q = np.empty(((m,n-1))) # (m, n-1)

    p = x[:m-1, :n] - x[1:m, :n]
    q = x[:m, :n-1] - x[:m, 1:n]

    return (p,q)


def proj_C(x, l, u): # e.g. proj B_lu
    # This can handle np.inf too
    return(np.clip(x, l, u))

def proj_P(p, q, tv_type='iso'):
    m = q.shape[0]
    n = p.shape[1]

    r = np.empty((m-1,n)) # (m-1) x n
    s = np.empty((m, n-1)) # m x (n-1)

    if tv_type == 'iso':
        r[:m-1, :n-1] = p[:m-1, :n-1] / np.maximum(1, np.sqrt(p[:m-1, :n-1]**2 + q[:m-1, :n-1]**2))
        r[:m-1, n-1] = p[:m-1, n-1] / np.maximum(1, np.abs(p[:m-1, n-1]))

        s[:m-1, :n-1] = q[:m-1, :n-1] / np.maximum(1, np.sqrt(p[:m-1, :n-1]**2 + q[:m-1, :n-1]**2))
        s[m-1, :n-1] = q[m-1, :n-1]  / np.maximum(1, np.abs(q[m-1, :n-1]))

    elif tv_type == 'l1':
        r = p / np.maximum(1, np.abs(p))
        s = q / np.maximum(1, np.abs(q))
    else:
        quit("Invalid Total Variation type (var: tv_type). Must be either 'iso' or 'l1'")

    return (r,s)


def padPSF(PSF, m, n):
    Pbig = np.zeros((m,n))
    Pbig[:PSF.shape[0], :PSF.shape[1]] = PSF
    return Pbig

# TODO: figure out what this is doing!!!!!
def dctshift(PSF, P_center):
    m, n = PSF.shape[:2]

    i = P_center[0]
    j = P_center[1]
    k = np.min([i, m-(i+1), j, n-(j+1)])

    PP = PSF[i-k:i+k+1, j-k:j+k+1]

    Z1 = np.diag(np.ones((k+1,)), k)
    Z2 = np.diag(np.ones((k,)), k+1)

    PP = (Z1 @ PP @ Z1.T) + (Z1 @ PP @ Z2.T) + (Z2 @ PP @ Z1.T) + (Z2 @ PP @ Z2.T) # @ is for matmul
    
    Ps = np.zeros((m,n))
    Ps[:PP.shape[0], :PP.shape[1]] = PP
    
    return Ps

# This only curr works for 2d!!
def circshift(PSF, num_pos):
    return np.roll(np.roll(PSF, num_pos[0], axis=0), num_pos[1], axis=1)