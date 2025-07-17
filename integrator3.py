from integrator import *

L = 5e-6
N = 100
layer_materials = ["FePS3"]*9
epsmax = 0.01 # %

mask_type = "image" # "square", "circle", "image", ""
path = "masks/meb_good100.png"



print(" > K bip boup...")
K = get_K_withgap(layer_materials, N, L, epsmax)

print(" > w bip boup...")
if mask_type == "":
    eigen_f, eigen_w = solv(K, 12, N)
else:
    match mask_type:
        case "square": mask_ind = square_mask_ind(N)
        case "circle": mask_ind = circular_mask_ind(N)
        case "image":  mask_ind = image_mask_ind(path)
    eigen_f, eigen_w = solv(K, 12, N, mask_ind)

np.savez("save3", eigen_f=eigen_f, eigen_w=eigen_w)

fig, ax = plt.subplots(3,4)
for i,w in enumerate(eigen_w):
    ax[i//4,i%4].set_title(f"({i+1}) f = {eigen_f[i]/1e6:{2}g} MHz")
    ax[i//4,i%4].imshow(w.T, origin='lower', cmap='twilight', extent=[0, L, 0, L], vmax=1, vmin=-1)
    ax[i//4,i%4].set_xlabel("x"); ax[i//4,i%4].set_ylabel("y")
plt.subplots_adjust(hspace=0.5)
plt.show()