import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from PIL import Image

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider



### - - - BUILDING K - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
### - - - __________ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def unVoigt(Lambda_ab):
    ind = {
        (0,0,0,0): (0,0),
        (1,1,0,0): (0,1), 
        (0,1,0,0): (0,5),
        (0,0,1,1): (1,0), 
        (1,1,1,1): (1,1),
        (0,1,1,1): (1,5),
        (0,0,0,1): (5,0),
        (1,1,0,1): (5,1),
        (0,1,0,1): (5,5)
    }
    Lambda_ijkl = np.zeros((2,2,2,2))
    for ijkl, ab in ind.items(): 
        Lambda_ijkl[*ijkl] = Lambda_ab[*ab]
    return Lambda_ijkl

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def getD_ijkl(Lambda_ijkl, h):
    return Lambda_ijkl*h**3/12

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def getT_klnm(N, L, T_max=1):
    T_klnm = np.zeros((2,2,N,N))
    for n in range(N):
        for m in range(N):
            x = ((n+0.5)/N-0.5)*L
            y = ((m+0.5)/N-0.5)*L
            r = np.sqrt(x**2+y**2)
            theta = np.arctan2(y,x)
            c, s = np.cos(theta), np.sin(theta)
            fact = T_max * r**2/(L/2)**2
            T_klnm[0,0,n,m] = c*c*fact
            T_klnm[0,1,n,m] = s*c*fact
            T_klnm[1,0,n,m] = s*c*fact
            T_klnm[1,1,n,m] = s*s*fact
    return T_klnm

def getN_ijnm(Lambda_ijkl, T_klnm, h):
    N = T_klnm.shape[2]
    N_ijnm = np.zeros((2,2,N,N))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    N_ijnm[i,j] += Lambda_ijkl[i,j,k,l]*T_klnm[k,l]*h
    return N_ijnm

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_zlayers(layer_types, h_dict, gap_dict):
    I = [0]
    for l1,l0 in zip(layer_types[1:],layer_types[:-1]):
        I.append(I[-1] + h_dict[l0])
        I.append(I[-1] + (gap_dict[l0]+gap_dict[l1])/2)
    I.append(I[-1] + h_dict[layer_types[-1]])
    return np.array(I) - I[-1]/2

def get_rhoh(layer_types, h_dict, gap_dict, rho_dict):
    rhoh = rho_dict[layer_types[0]] * ( h_dict[layer_types[0]] + gap_dict[layer_types[0]]/2 )
    for l in layer_types[1:-1]:
        rhoh += rho_dict[l] * ( h_dict[l] + gap_dict[l] )
    rhoh += rho_dict[layer_types[-1]] * ( h_dict[layer_types[-1]] + gap_dict[layer_types[-1]]/2 )
    return rhoh

def getA_ijkl(layer_types, Lambda_ijkl_dict, h_dict, gap_dict):
    I = get_zlayers(layer_types, h_dict, gap_dict)
    A_ijkl = np.zeros((2,2,2,2))
    for i,l in enumerate(layer_types):
        A_ijkl += Lambda_ijkl_dict[l] * (I[i+1]-I[i])
    return A_ijkl

def getC_ijkl(layer_types, Lambda_ijkl_dict, h_dict, gap_dict):
    I = get_zlayers(layer_types, h_dict, gap_dict)
    C_ijkl = np.zeros((2,2,2,2))
    for i,l in enumerate(layer_types):
        C_ijkl += Lambda_ijkl_dict[l]/3 * (I[i+1]**3-I[i]**3)
    return C_ijkl

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def deriv0(N, dx=1):
    return sp.eye(N)
def deriv1(N, dx=1):
    diagonals = [-np.ones(N-1), np.ones(N-1)]
    offsets = [1, -1]
    D = sp.diags(diagonals, offsets, shape=(N, N), format='csr')
    # D[0,0] = -2; D[0,2] = 2; D[-1,-1] = 2; D[-1,-2] = -2        <- what we would need to take indot account the boundaries,
    return D/2/dx                                                  # but it is not that important...
def deriv2(N, dx=1):
    diagonals = [np.ones(N-1), -2*np.ones(N), np.ones(N-1)]
    offsets = [1, 0, -1]
    return sp.diags(diagonals, offsets, shape=(N, N), format='csr')/dx**2
def deriv3(N, dx=1):
    diagonals = [np.ones(N-2), -2*np.ones(N-1), 2*np.ones(N-1), -np.ones(N-2)]
    offsets = [-2, -1, 1, 2]
    return sp.diags(diagonals, offsets, shape=(N, N), format='csr')/8/dx**3
def deriv4(N, dx=1):
    diagonals = [np.ones(N-2), -4*np.ones(N-1), 6*np.ones(N), -4*np.ones(N-1), np.ones(N-2)]
    offsets = [-2, -1, 0, 1, 2]
    return sp.diags(diagonals, offsets, shape=(N, N), format='csr')/dx**4

def Deriv(nx, ny, N, dx=1):
    deriv = [deriv0,deriv1,deriv2,deriv3,deriv4]
    return sp.kron(deriv[nx](N,dx),deriv[ny](N,dx))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def buildK_unstressed_homogeneous(N, L, Lambda_ab, rho, h):
    dx = L/N
    Lambda_ijkl = unVoigt(Lambda_ab)
    D_ijkl = getD_ijkl(Lambda_ijkl, h)
    K = sp.csr_matrix((N**2, N**2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    K += D_ijkl[i,j,k,l] * Deriv((4-i-j-k-l),(i+j+k+l), N,dx)
    return K/(rho*h)

def buildK_stressed_homogeneous(N, L, Lambda_ab, rho, h, T_max=1):
    dx = L/N
    Lambda_ijkl = unVoigt(Lambda_ab)
    T_klnm = getT_klnm(N, L, T_max)
    N_ijnm = getN_ijnm(Lambda_ijkl, T_klnm, h)
    D_ijkl = getD_ijkl(Lambda_ijkl, h)
    dNinm = np.gradient(N_ijnm[:,0], axis=1) + np.gradient(N_ijnm[:,1], axis=2) # d0Ni0nm + d1Ni1nm
    K = sp.csr_matrix((N**2, N**2), dtype=np.float64)
    for i in range(2):
        K -= 0.5 * sp.diags(dNinm[i].flatten()) @ Deriv((1-i),(i), N,dx)
        for j in range(2):
            K -= 0.5 * sp.diags(N_ijnm[i,j].flatten()) @ Deriv((2-i-j),(i+j), N,dx)
            for k in range(2):
                for l in range(2):
                    K += D_ijkl[i,j,k,l] * Deriv((4-i-j-k-l),(i+j+k+l), N,dx)
    return K/(rho*h)

def buildK_multilayered_stressed(N, L, layer_types, Lambda_ab_dict, rho_dict, h_dict, gap_dict, T_max=1):
    dx = L/N
    Lambda_ijkl_dict = {t:unVoigt(Lambda_ab) for t,Lambda_ab in Lambda_ab_dict.items()}
    rhoh = get_rhoh(layer_types, h_dict, gap_dict, rho_dict)
    A_ijkl = getA_ijkl(layer_types, Lambda_ijkl_dict, h_dict, gap_dict)
    C_ijkl = getC_ijkl(layer_types, Lambda_ijkl_dict, h_dict, gap_dict)
    T_klnm = getT_klnm(N, L, T_max)
    djT_klnm = np.array([np.gradient(T_klnm, axis=2), np.gradient(T_klnm, axis=3)])
    K = sp.csr_matrix((N**2, N**2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    K += 0.5 * A_ijkl[i,j,k,l] * ( sp.diags(djT_klnm[j,k,l].flatten()) @ Deriv((1-i),(i), N,dx)
                                                     + sp.diags(T_klnm[k,l].flatten()) @ Deriv((2-i-j),(i+j), N,dx) )
                    K += C_ijkl[i,j,k,l] * Deriv((4-i-j-k-l),(i+j+k+l), N,dx)
    return K/rhoh



### - - - SOLVING K - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
### - - - _________ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def getMask_nm_circle(N):
    mask = np.ones((N,N), dtype=bool)
    for n in range(N):
        for m in range(N):
            if 4*( ((n+.5)/(N)-.5)**2 + ((m+.5)/(N)-.5)**2 ) > 1 : 
                mask[n,m] = 0
    return np.where(mask.ravel())[0]

def getMask_nm_image(path):
    mask = ((np.array(Image.open(path).convert('L').transpose(Image.ROTATE_270))) > 128)
    return np.where(mask.ravel())[0]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solvK(K, n_modes, N, mask_nm=None, which='SA', **eigsh_param):
    '''-> eigen_f, eigen_z'''
    if mask_nm is None : 
        eigen_wl, eigen_z = eigsh(K, k=n_modes, which=which, **eigsh_param)
        return np.abs(eigen_wl)**0.5/(2*np.pi), np.array([eigen_z[:,i].reshape((N,N))/np.max(np.abs(eigen_z[:,i])) for i in range(n_modes)])
    Kr = K[mask_nm][:, mask_nm]
    eigen_wl, eigen_zr = eigsh(Kr, k=n_modes, which=which, **eigsh_param)
    eigen_z = []
    for i in range(n_modes):
        z = np.zeros(N**2)
        z[mask_nm] = eigen_zr[:,i]
        eigen_z.append(z.reshape((N,N)))
    return np.abs(eigen_wl)**0.5/(2*np.pi), np.array([np.array(zn)/np.max(np.abs(zn)) for zn in eigen_z])



### - - - MOTION - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
### - - - ______ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def anim(fnl, wnl, time_step, L, N, rhoh=1, Al=1):
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax_freq = plt.axes([0.25, 0.03, 0.5, 0.03])

    def cnl(freq): return Al/(abs((freq**2-fnl**2)+1e-10))/(2*np.pi)
    Z = np.sum([wn*cn for wn,cn in zip(wnl,cnl(fnl[0]))], axis=0)
    Zmax = np.max(Z)
    surf = [ax.plot_surface(X, Y, Z, cmap='twilight', vmax=Zmax*0.8, vmin=-Zmax*0.8)]

    slider_freq = Slider(ax_freq, 'LASER frequency', 0.7*fnl[0], fnl[-1]*1.05, valinit=fnl[0])

    def update(frame):
        ax.clear()
        freqs = slider_freq.val
        w = np.sum([wn*cn for wn,cn in zip(wnl,cnl(freqs))], axis=0)
        Z = w*np.cos((2*np.pi*freqs*time_step*frame/30)%(2*np.pi))
        Zmax = np.max(w)
        ax.set_zlim(-1.5*Zmax, 1.5*Zmax)
        surf[0] = ax.plot_surface(X, Y, Z, cmap='twilight', vmax=Zmax*0.8, vmin=-Zmax*0.8)
        return surf

    ani = FuncAnimation(fig, update, frames=1000, interval=50, blit=False)

    plt.show()