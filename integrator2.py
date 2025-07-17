from integrator import *

L = 5e-6
N = 100
n_layers = 13
material = "FePS3"
rho = rho_val[material]
h = 6.722e-10 * n_layers
epsmax = 0.0 # %

n_modes = 12

Lambdaijkl = convert_ijkl(Lambda_val[material])
Dijkl = Lambdaijkl*h**3/12

Nij = get_Nijxy(Lambdaijkl, N, h, L, epsmax)

mask_type = "image" # "square", "circle", "image", ""
path = "masks/meb_good100.png"



print(" > K bip boup...")
K = get_K_noncstN(Nij, Dijkl, N,L,rho*h)

print(" > w bip boup...")
if mask_type == "":
    eigen_f, eigen_w = solv(K, n_modes, N)
else:
    match mask_type:
        case "square": mask_ind = square_mask_ind(N)
        case "circle": mask_ind = circular_mask_ind(N)
        case "image":  mask_ind = image_mask_ind(path)
    eigen_f, eigen_w = solv(K, n_modes, N, mask_ind)


fig, ax = plt.subplots(3,4)
for i,w in enumerate(eigen_w):
    ax[i//4,i%4].set_title(f"({i+1}) f = {eigen_f[i]/1e6:{2}g} MHz")
    ax[i//4,i%4].imshow(w.T, origin='lower', cmap='twilight', extent=[0, L, 0, L], vmax=1, vmin=-1)
    ax[i//4,i%4].set_xlabel("x"); ax[i//4,i%4].set_ylabel("y")
plt.subplots_adjust(hspace=0.5)
plt.show()

#for i,w in enumerate(eigen_w):
#    vmax = max(np.max(w), -np.min(w))
#    fig, ax = plt.subplots()
#    fig.suptitle(f"Mode {i+1} : f = {abs(eigen_wl[i])**0.5/(2*np.pi):{2}g} Hz")
#    im = ax.imshow(w.T, origin='lower', cmap='twilight', extent=[0, L, 0, L], vmax=vmax, vmin=-vmax)
#    cbar = fig.colorbar(im, ax=ax, label='w(x, y) (a.u., should be normalized)')
#    ax.set_xlabel("x"); ax.set_ylabel("y")
#    fig.tight_layout()
#    plt.show()
#    #fig.savefig(f"save/meb_good150/mode_{i:03}.jpg")

