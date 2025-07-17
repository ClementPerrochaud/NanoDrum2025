from integrator import *


L = 5e-6
N = 200
rho = 2.94 * 1000
h = 6.722e-10 * 12e3

n_modes = 100

Lambdaijkl = convert_ijkl(Lambda_val["FePS3"])
Dijkl = Lambdaijkl*h**3/12

Nij   = np.zeros((2,2))
Nij[0,0], Nij[1,1] = 1, 1
Nij *= h

path1 = "masks/meb_good200.png"
path2 = "masks/meb_perfect200.png"



print(" > K  bip boup...")
K = get_K(Nij, Dijkl, N, L, rho*h)

print(" > w1 bip boup...")
mask1_ind = image_mask_ind(path1)
eigen_f1, eigen_w1 = solv(K, n_modes, N, mask1_ind)

print(" > w2 bip boup...")

mask2_ind = image_mask_ind(path2)
eigen_f2, eigen_w2 = solv(K, n_modes, N, mask2_ind)

  

for i,(w1,w2) in enumerate(zip(eigen_w1,eigen_w2)):
    fig, (ax1,ax2) = plt.subplots(1,2)
    fig.suptitle(f"Mode {i+1} :")
    ax1.set_title(f"f = {eigen_f1[i]:{4}g} Hz")
    ax2.set_title(f"f = {eigen_f2[i]:{4}g} Hz")
    im1 = ax1.imshow(w1.T, origin='lower', cmap='twilight', extent=[0, L, 0, L], vmax=1, vmin=-1)
    im2 = ax2.imshow(w2.T, origin='lower', cmap='twilight', extent=[0, L, 0, L], vmax=1, vmin=-1)
    #cbar = fig.colorbar(im2, ax=(ax1,ax2), label='w(x, y)', orientation='horizontal', location='bottom') # elle marche pas c'est chiant
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax2.set_xlabel("x"); ax2.set_ylabel("y")
    fig.tight_layout()
    #plt.show()
    fig.savefig(f"save/good_vs_perfect/mode_{i:04}.jpg")