import os
import numpy as np
from models import RolemodelLearning
from plot_function import subplots_horizontal
from maths import generate_structure


epsilon = 1
lam = 0.9
r_cutoff = 1.7
r_center = 1.1
eta = 100
r_min = 1

natoms = 13
global_mins = np.load('global_min_3d.npy')
global_minimum = global_mins[np.where(global_mins[:, 0] == natoms), 1] * epsilon

rattle_percent = 40
rattle_factor = rattle_percent / 100

num_clusters = 5
num_rolemodels = 5

parameters = {
    'r_min': r_min,
    'epsilon': epsilon,
    'eta': eta,
    'lam': lam,
    'r_center': r_center,
    'r_cutoff': r_cutoff,
    'global_minimum': global_minimum,
    'rattle_factor': rattle_factor
}

max_iter = 100000
niter = 1000
atol = 1e-02

ndims = 3

N = 100
##########

if ndims == 2:
    scale = 5
elif ndims == 3:
    scale = 3
else:
    print("What scale should i use? :(")
    raise ValueError("Scale not defined")

name = 'iter_{0}atoms{1}d_{2}rm_{3}rattle.npy'.format(natoms, ndims, num_rolemodels, rattle_percent)

if not os.path.isfile(name):
    np.save(name, [])

debug = False
silent = False
plotting = False

for iteration in range(N):
    print("Run number {}".format(iteration))
    structure = generate_structure(natoms, scale=scale, ndims=ndims)
    model = RolemodelLearning(
        structure, num_clusters, num_rolemodels, niter, parameters, max_iter, atol,
        debug=debug, silent=silent
    )
    model.run_model()

    if plotting is True:

        energies = model.give_energies
        distances = model.give_distances
        title = 'Energy and cluster distance as a function of iterations (time)'
        subplots_horizontal(
            energies[1:], distances[1:], title, first_ylabel='Energy',
            second_ylabel='Cluster distance', xlabel='Iteration'
        )
        title = 'Energy and cluster distance minima as a function of iterations (time)'
        subplots_horizontal(
            energies[1::2], distances[2::2], title, first_ylabel='Energy',
            second_ylabel='Cluster distance', xlabel='Iteration'
        )

    itera = model.give_iteration
    if itera is not None:
        iterations = np.append(np.load(name), itera)
        np.save(name, iterations)
