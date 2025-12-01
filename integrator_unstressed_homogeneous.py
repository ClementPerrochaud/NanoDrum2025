from integrator import *
import data

L = 5e-6
N = 100
n_layers = 10

rho = data.rho_dict["NiPS3"]
h = data.h_dict["NiPS3"] * n_layers
Lambda_ab = data.Lambda_ab_dict["NiPS3"]

mask = "image" # or "circle", or "None"
path = f"masks/meb_good{N}.png"


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


print(" > K bip-boup...")
K = buildK_unstressed_homogeneous(N,L,Lambda_ab,rho,h)

print(" > z bip-boup...")
match mask:
    case "circle": mask_nm = getMask_nm_circle(N)
    case "image":  mask_nm = getMask_nm_image(path)
    case _:        mask_nm = None
eigen_f, eigen_z = solvK(K, 15, N, mask_nm)

np.savez("eigensave", eigen_f=eigen_f, eigen_z=eigen_z)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


fig, ax = plt.subplots(3,5, figsize=(18, 10))
for i,w in enumerate(eigen_z):
    ax[i//5,i%5].set_title(f"({i+1}) f = {eigen_f[i]/1e6:{2}g} MHz")
    ax[i//5,i%5].imshow(w.T, origin='lower', cmap='twilight', extent=[0, L, 0, L], vmax=1, vmin=-1)
    ax[i//5,i%5].set_xlabel("x"); ax[i//5,i%5].set_ylabel("y")
plt.subplots_adjust(hspace=0.5)

plt.show()
