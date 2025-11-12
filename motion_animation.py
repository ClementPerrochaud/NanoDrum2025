from integrator import *
import data

L = 5e-6
N = 100
layer_materials = ["NiPS3"]*10

time_step = 1e-7
A_laser = 1e-6

pixel_size = 4

load = np.load("eigensave.npz")
eigen_f, eigen_w = load["eigen_f"], load["eigen_w"]

rhoh = get_rhoh(layer_materials, *data.dicts_rhoh)
anim(eigen_f, eigen_w[:,::pixel_size,::pixel_size], time_step, L,N//pixel_size, rhoh, A_laser)