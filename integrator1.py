from integrator import *


L = 5e-6
N = 70
n_layers = 10
rho = 2.94e3
h = 6.722e-10 * n_layers

n_modes = 30

Lambdaijkl = convert_ijkl(Lambda_val["FePS3"])
Dijkl = Lambdaijkl*h**3/12

Nij   = np.zeros((2,2))
#Nij[0,0], Nij[1,1] = 1, 1
#Nij *= h

mask_type = "circle" # "square", "circle", "image", ""
path = "masks/meb_bad100.png"



print(" > K bip boup...")
K = get_K(Nij, Dijkl, N, L, rho*h)

print(" > w bip boup...")
if mask_type == "":
    eigen_f, eigen_w = solv(K, n_modes, N)
else :
    match mask_type:
        case "square": mask_ind = square_mask_ind(N)
        case "circle": mask_ind = circular_mask_ind(N)
        case "image":  mask_ind = image_mask_ind(path)
    eigen_f, eigen_w = solv(K, n_modes, N, mask_ind)



for i,w in enumerate(eigen_w):
    fig, ax = plt.subplots()
    fig.suptitle(f"Mode {i+1} : f = {eigen_f[i]:{4}g} Hz")
    im = ax.imshow(w.T, origin='lower', cmap='twilight', extent=[0, L, 0, L], vmax=1, vmin=-1)
    cbar = fig.colorbar(im, ax=ax, label='w(x, y) (arbitrary units, should be normalized)')
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    plt.show()
    #fig.savefig(f"save/meb_good200/mode_{i:04}.jpg")