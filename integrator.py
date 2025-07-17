import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider


def convert_ijkl(Lambda_val):
    ind = {
    (0,0,0,0): (0,0),
    (1,1,0,0): (0,1), 
    (0,1,0,0): (0,5),
    (0,0,1,1): (1,0), 
    (1,1,1,1): (1,1),
    (0,1,1,1): (1,5),
    (0,0,0,1): (5,0),
    (1,1,0,1): (5,1),
    (0,1,0,1): (5,5)}
    val = {k:Lambda_val[ind[k]] for k in ind}
    Lambda_val = np.zeros((2,2,2,2))
    for k in val: Lambda_val[*k] = val[k]
    return Lambda_val


Lambda_val = {
    "FePS3" : np.array([
    [120.5, 34.1,  5.7,   0.85,   0.0,   0.0],
    [34.1, 119.8,  5.3,  -1.55,   0.0,   0.0],
    [5.7,   5.3,  28.4,  -0.05,   0.0,   0.0],
    [0.85, -1.55, -0.05,  1.18,   0.0,   0.0],
    [0.0,   0.0,   0.0,   0.0,    9.88, -1.05],
    [0.0,   0.0,   0.0,   0.0,   -1.05,  1.68]])*1e9,

    "NiPS3" : np.array([
    [124.0, 35.1,  5.9,   0.87,   0.0,   0.0],
    [35.1, 123.3,  5.5,  -1.60,   0.0,   0.0],
    [5.9,   5.5,  29.2,  -0.05,   0.0,   0.0],
    [0.87, -1.60, -0.05,  1.21,   0.0,   0.0],
    [0.0,   0.0,   0.0,   0.0,   10.2,  -1.08],
    [0.0,   0.0,   0.0,   0.0,   -1.08,  1.73]])*1e9,

    "hBN" : np.array([
    [865. , 150. ,   0. ,   0. ,   0. ,   0. ],
    [150. , 865. ,   0. ,   0. ,   0. ,   0. ],
    [  0. ,   0. ,  38.7,   0. ,   0. ,   0. ],
    [  0. ,   0. ,   0. ,   4.95,  0. ,   0. ],
    [  0. ,   0. ,   0. ,   0. ,   4.95,  0. ],
    [  0. ,   0. ,   0. ,   0. ,   0. , 357.5]])*1e9,

    "graphene" :  np.array([
    [1020., 175.,   0.,   0.,   0.,   0.],
    [175., 1020.,   0.,   0.,   0.,   0.],
    [  0.,   0.,   36.,   0.,   0.,   0.],
    [  0.,   0.,    0., 170.,  0.,   0.],
    [  0.,   0.,    0.,   0., 170.,  0.],
    [  0.,   0.,    0.,   0.,   0., 425.]])*1e9,

    "CrCl3" : np.array([
    [66.24, 16.56,  0.,    0.,    0.,    0.  ],
    [16.56, 66.24,  0.,    0.,    0.,    0.  ],
    [ 0.,    0.,   12.,    0.,    0.,    0.  ],
    [ 0.,    0.,    0.,   24.84,  0.,    0.  ],
    [ 0.,    0.,    0.,    0.,   24.84,  0.  ],
    [ 0.,    0.,    0.,    0.,    0.,   49.68]])*1e9,

    "WS2" : np.array([
    [236.,  53.,   0.,  0.,  0.,  0.],
    [ 53., 236.,   0.,  0.,  0.,  0.],
    [  0.,   0.,  44.,  0.,  0.,  0.],
    [  0.,   0.,   0.,  42., 0.,  0.],
    [  0.,   0.,   0.,   0., 42., 0.],
    [  0.,   0.,   0.,   0.,  0., 91.5]])*1e9
}

rho_val = {
    "FePS3":    2.94e3,
    "NiPS3":    3.18e3,
    "hBN":      2.1e3,
    "graphene": 2.267e3,
    "CrCl3":    2.87e3,
    "WS2":      7.5e3
}

h_val = {
    "FePS3":    6.722e-10,
    "NiPS3":    6.32e-10,
    "hBN":      3.3e-10,
    "graphene": 3.3e-10,
    "CrCl3":    6.3e-10,
    "WS2":      6.2e-10
}

gap_val = {
    "FePS3":    3.25e-10,
    "NiPS3":    3.26e-10,
    "hBN":      3.3e-10,
    "graphene": 3.3e-10,
    "CrCl3":    6e-10,
    "WS2":      3.3e-10
}


def get_layer_z(layer_types, h_val=h_val, gap_val=gap_val):
    I = [0]
    for l1,l0 in zip(layer_types[1:],layer_types[:-1]):
        I.append(I[-1] + h_val[l0])
        I.append(I[-1] + (gap_val[l0]+gap_val[l1])/2)
    I.append(I[-1] + h_val[layer_types[-1]])
    return np.array(I) - I[-1]/2


def get_rhoh(layer_types, h_val=h_val, gap_val=gap_val, rho_val=rho_val):
    rhoh = rho_val[layer_types[0]]*( h_val[layer_types[0]] + gap_val[layer_types[0]]/2 )
    for l in layer_types[1:-1]:
        rhoh += rho_val[l]*( h_val[l] + gap_val[l] )
    rhoh += rho_val[layer_types[-1]]*( h_val[layer_types[-1]] + gap_val[layer_types[-1]]/2 )
    return rhoh


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def get_Nijxy(Lambdaijkl, N, h=1, L=1, epsmax=1):
    Nijxy = np.zeros((2,2,N,N))
    for ix in range(N):
        x = ((ix+0.5)/N-0.5)*L
        for iy in range(N):
            y = ((iy+0.5)/N-0.5)*L
            r = np.sqrt(x**2+y**2)
            theta = np.atan2(y,x)
            e0 = (r**2/L**2)*4*h*epsmax/100
            cos, sin = np.cos(theta), np.sin(theta)
            N1 = (Lambdaijkl[0,0,0,0]*cos**2 + Lambdaijkl[0,0,1,1]*sin**2)*e0
            N2 = (Lambdaijkl[0,0,1,1]*cos**2 + Lambdaijkl[1,1,1,1]*sin**2)*e0
            N3 =  Lambdaijkl[0,1,0,1]*sin**2*e0
            Nijxy[0,0,ix,iy] = N1
            Nijxy[0,1,ix,iy] = N3
            Nijxy[1,0,ix,iy] = N3
            Nijxy[1,1,ix,iy] = N2
    return Nijxy


def get_eijxy(N, L=1, epsmax=1):
    eijxy = np.zeros((2,2,N,N))
    for ix in range(N):
        x = ((ix+0.5)/N-0.5)*L
        for iy in range(N):
            y = ((iy+0.5)/N-0.5)*L
            r = np.sqrt(x**2+y**2)
            theta = np.atan2(y,x)
            e0 = (r**2/L**2)*4*epsmax/100
            cos, sin = np.cos(theta), np.sin(theta)
            eijxy[0,0,ix,iy] = cos**2*e0
            eijxy[0,1,ix,iy] = sin*cos*e0
            eijxy[1,0,ix,iy] = sin*cos*e0
            eijxy[1,1,ix,iy] = sin**2*e0
    return eijxy


def get_Aijkl_Cijkl(layer_types, Lambda_val=Lambda_val, h_val=h_val, gap_val=gap_val):
    I = get_layer_z(layer_types, h_val, gap_val)
    Aijkl, Cijkl = np.zeros((2,2,2,2)), np.zeros((2,2,2,2))
    save = {}
    for i,l in enumerate(layer_types):
        if not l in save: save[l] = convert_ijkl(Lambda_val[l])
        Lijkl = save[l]
        Aijkl += Lijkl*(I[i+1]-I[i])
        Cijkl += Lijkl*(I[i+1]**3-I[i]**3)
    return Aijkl, Cijkl


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def __deriv0(N, dx=1):
    return sp.eye(N)
def __deriv1(N, dx=1):
    diagonals = [-np.ones(N-1), np.ones(N-1)]
    offsets = [1, -1]
    D = sp.diags(diagonals, offsets, shape=(N, N), format='csr')
    #D[0,0] = -2; D[0,2] = 2; D[-1,-1] = 2; D[-1,-2] = -2
    return D/2/dx
def __deriv2(N, dx=1):
    diagonals = [np.ones(N-1), -2*np.ones(N), np.ones(N-1)]
    offsets = [1, 0, -1]
    return sp.diags(diagonals, offsets, shape=(N, N), format='csr')/dx**2
def __deriv3(N, dx=1):
    diagonals = [np.ones(N-2), -2*np.ones(N-1), 2*np.ones(N-1), -np.ones(N-2)]
    offsets = [-2, -1, 1, 2]
    return sp.diags(diagonals, offsets, shape=(N, N), format='csr')/2/dx**3
def __deriv4(N, dx=1):
    diagonals = [np.ones(N-2), -4*np.ones(N-1), 6*np.ones(N), -4*np.ones(N-1), np.ones(N-2)]
    offsets = [-2, -1, 0, 1, 2]
    return sp.diags(diagonals, offsets, shape=(N, N), format='csr')/dx**4

def Deriv(nx, ny, N, dx=1):
    deriv = [__deriv0,__deriv1,__deriv2,__deriv3,__deriv4]
    return sp.kron(deriv[nx](N,dx),deriv[ny](N,dx))


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def get_K(Nij, Dijkl, N, L=1, rhoh=1):
    dx = L/N
    K = sp.csr_matrix((N**2, N**2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            K += 0.5 * Nij[i,j] * Deriv((2-i-j),(i+j), N,dx)
            for k in range(2):
                for l in range(2):
                    K -= Dijkl[i,j,k,l] * Deriv((4-i-j-k-l),(i+j+k+l), N,dx)
    return K/rhoh


def get_K_noncstN(Nijxy, Dijkl, N, L=1, rhoh=1):
    dx = L/N
    dNixy = np.gradient(Nijxy[:,0], axis=1) + np.gradient(Nijxy[:,1], axis=2) # d0Ni0xy + d1Ni1xy
    K = sp.csr_matrix((N**2, N**2), dtype=np.float64)
    for i in range(2):
        K += 0.5 * sp.diags(dNixy[i].flatten()) @ Deriv((1-i),(i), N,dx)
        for j in range(2):
            K += 0.5 * sp.diags(Nijxy[i,j].flatten()) @ Deriv((2-i-j),(i+j), N,dx)
            for k in range(2):
                for l in range(2):
                    K -= Dijkl[i,j,k,l] * Deriv((4-i-j-k-l),(i+j+k+l), N,dx)
    return K/rhoh


def get_K_withgap(layer_types, N, L=1, epsmax=0, Lambda_val=Lambda_val, h_val=h_val, rho_val=rho_val, gap_val=gap_val):
    dx = L/N
    rhoh = get_rhoh(layer_types, h_val, gap_val, rho_val)
    Aijkl, Cijkl = get_Aijkl_Cijkl(layer_types, Lambda_val, h_val, gap_val)
    eklxy = get_eijxy(N,L,epsmax)
    djeklxy = np.array([np.gradient(eklxy, axis=2), np.gradient(eklxy, axis=3)])
    K = sp.csr_matrix((N**2, N**2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    K += 0.5 * Aijkl[i,j,k,l] * ( sp.diags(djeklxy[j,k,l].flatten()) @ Deriv((1-i),(i), N,dx)
                                                    + sp.diags(eklxy[k,l].flatten()) @ Deriv((2-i-j),(i+j), N,dx) )
                    K += Cijkl[i,j,k,l] * Deriv((4-i-j-k-l),(i+j+k+l), N,dx)
    return K/rhoh


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def square_mask_ind(N, n=1):
    mask = np.ones((N,N), dtype=bool)
    for i in range(n):
        mask[n, :], mask[-1-n, :], mask[:, n], mask[:, -1-n] = (False,)*4
    return np.where(mask.ravel())[0]


def circular_mask_ind(N):
    mask = np.ones((N,N), dtype=bool)
    for i in range(N):
        for j in range(N):
            if 4*( ((i+.5)/(N)-.5)**2 + ((j+.5)/(N)-.5)**2 ) > 1 : 
                mask[i,j] = 0
    return np.where(mask.ravel())[0]


def image_mask_ind(path):
    mask = ((np.array(Image.open(path).convert('L').transpose(Image.ROTATE_270))) > 128)
    return np.where(mask.ravel())[0]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def solv(K, n_modes, N, mask_ind=None, which='SA', **eigsh_param):
    '''-> eigen_f, eigen_w'''
    if mask_ind is None : 
        eigen_wl, eigen_w = eigsh(K, k=n_modes, which=which, **eigsh_param)
        return np.abs(eigen_wl)**0.5/(2*np.pi), np.array([eigen_w[:,i].reshape((N,N))/np.max(np.abs(eigen_w[:,i])) for i in range(n_modes)])
    Kr = K[mask_ind][:, mask_ind]
    eigen_wl, eigen_wr = eigsh(Kr, k=n_modes, which=which, **eigsh_param)
    eigen_w = []
    for i in range(n_modes):
        w = np.zeros(N**2)
        w[mask_ind] = eigen_wr[:,i]
        eigen_w.append(w.reshape((N,N)))
    return np.abs(eigen_wl)**0.5/(2*np.pi), np.array([np.array(eigen_wn)/np.max(np.abs(eigen_wn)) for eigen_wn in eigen_w])


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def anim(fnl, wnl, time_step, L, N, rhoh=1, Al=1):
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax_freq = plt.axes([0.25, 0.1, 0.5, 0.03])

    def cnl(freq): return Al/(abs((freq**2-fnl**2)+1e-10))/(2*np.pi)
    Z = np.sum([wn*cn for wn,cn in zip(wnl,cnl(fnl[0]))], axis=0)
    Zmax = np.max(Z)
    surf = [ax.plot_surface(X, Y, Z, cmap='twilight', vmax=Zmax*0.8, vmin=-Zmax*0.8)]

    slider_freq = Slider(ax_freq, 'fréquence laser', 0.7*fnl[0], fnl[-1]*1.05, valinit=fnl[0])

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