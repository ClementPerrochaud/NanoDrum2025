from integrator import *

# tout comme integrator3

L = 5e-6
N = 100
layer_materials = ["FePS3"]*9

mask_type = "image" # "square", "circle", "image", ""
path = "masks/meb_good100.png"


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

time_step = 1e-7
A_laser = 1e-6

load = np.load("save3.npz")
eigen_f, eigen_w = load["eigen_f"], load["eigen_w"]

rhoh = get_rhoh(layer_materials)
anim(eigen_f, eigen_w[:,::4,::4], time_step, L,N//4, rhoh, A_laser)