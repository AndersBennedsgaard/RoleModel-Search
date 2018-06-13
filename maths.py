import numpy as np
import math


def compute_euclidian_distances(xyz, option='sqrt'):
    m, n = xyz.shape
    g = np.dot(xyz, xyz.T)  # Gram matrix
    h = np.tile(np.diag(g), (m, 1))
    d = h + h.T - 2 * g
    if option == 'sqrt':
        d = np.sqrt(d)
    elif option != '' or option != 'squared':
        raise TypeError("option can be 'sqrt' or '' ")
    return d


def compute_sizes(coordinates):
    """
    All sizes of the structure is calculated, using Pythagoras's theorem
    :param coordinates: This should be all the coordinates of a structure, with vertical positioned x-, y-, and z-arrays
    :return: A vertical numpy-array with N sizes, where N is the number of atoms in the structure
    """
    sizes = np.sqrt(np.sum(coordinates ** 2, axis=1))[np.newaxis].T
    return sizes


def normalize_vectors(xyz):
    dists = compute_sizes(xyz)
    if np.any(dists == 0):
        raise ValueError("Division by zero!")
    normalized = xyz / dists
    return normalized


def remove_diagonal(d):
    r = d[~np.eye(d.shape[0], dtype=bool)].reshape(d.shape[0], -1)
    return r


def cutoff_function(structure, r_cutoff):
    f_c = 0.5 * (1 + np.cos(math.pi * structure / r_cutoff))
    f_c[structure > r_cutoff] = 0
    return f_c


def gradient_cutoff(structure, r_cutoff):
    g_c = math.pi/2/r_cutoff * np.sin(structure*math.pi/r_cutoff)
    g_c[structure > r_cutoff] = 0
    return g_c


def split_seq(seq, num_pieces):
    newseq = []
    splitsize = 1.0 / num_pieces * len(seq)
    for i in range(num_pieces):
        newseq.append(seq[int(round(i * splitsize)):int(round((i + 1) * splitsize))])
    return newseq


def sigmoid(x):
    np.seterr(over='ignore')
    sigma = 1 / (1 + np.exp(-x))
    return sigma


def calculate_features(structure, lam, eta, r_center, r_cutoff, upscale=1):
    r_ij = compute_euclidian_distances(structure)
    r_ij = remove_diagonal(r_ij)
    f_c = cutoff_function(r_ij, r_cutoff)
    rho1 = upscale * np.sum(np.exp(-r_ij / lam) * f_c, axis=1)
    rho2 = np.sum(np.exp(-eta * ((r_ij - r_center) / r_cutoff) ** 2) * f_c, axis=1)

    features = np.append(rho1, rho2).reshape(2, r_ij.shape[0]).T

    return features


def lennard_jones(structure, r_min=1, epsilon=1):
    """
    Computes the Lennard Jones energy for N atoms

    :param structure: The structure of the atoms, given as a N*3-numpy array with x-, y- and z-coordinates.
    :param r_min: The distance two atoms are at lowest energy (epsilon)
    :param epsilon: The depth of the energy-well
    :return: The energy calculated using a Lennard Jones potential
    """

    dists = compute_euclidian_distances(structure)
    dists = remove_diagonal(dists)
    energy = epsilon/2 * np.sum((r_min/dists) ** 12 - 2 * (r_min/dists) ** 6)
    return energy


def cluster_distance(features, centroids, option='min'):
    """
    Computes the feature-to-centroid distance, using Pythagoras's theorem.
    The smallest feature-to-centroid distance for each atom is used, and summed up.

    :param features:
    :param centroids: Centroids calculated with a clustering method, for example the K-Means method
    :param option: Option can be either 'min' or 'list'. If 'list' is specified, a list of minimum is returned
    :return: The total structure-to-centroid distance or a 1D-list of minimum, depending on the option specified
    """
    if option is not 'min' and option is not 'list':
        print(option)
        raise ValueError("option must be either 'min' or 'list'.")

    num_centroids, num_features_centroids = centroids.shape
    num_atoms, num_features = features.shape
    if num_features != num_features_centroids:
        raise ValueError("The number of features in both the 'features'- and 'centroids'-arguments must be equal.")

    cluster_dist = np.zeros([num_atoms, num_centroids])
    for i in range(num_centroids):
        cluster_dist[:, i] = compute_sizes(features - centroids[i, :]).flatten()

    try:
        minimum = np.amin(cluster_dist, axis=1)
    except ValueError:
        print(features)
        print(centroids)
        print(num_atoms)
        print(num_centroids)
        raise ValueError("Error")

    if option is 'min':
        cluster_dist = np.sum(minimum)
    else:
        minimum_index = np.argmin(cluster_dist, axis=1)
        cluster_dist = (minimum, minimum_index)

    return cluster_dist


def gradient_lennard_jones(structure, r_min, epsilon=1):
    r_ij = remove_diagonal(compute_euclidian_distances(structure))
    num_atoms, ndims = structure.shape

    g_lj = np.zeros([num_atoms, ndims])
    for i in range(ndims):
        xyz = structure[:, i][np.newaxis].T
        xyz_ij = remove_diagonal(xyz.T - xyz)
        g = 12 * epsilon / (r_min ** 2) * np.sum(((r_min / r_ij) ** 14 - (r_min / r_ij) ** 8) * xyz_ij, axis=1)
        g_lj[:, i] = g.copy()

    return g_lj


def relax_structure(structure, r_min=1, epsilon=1, max_iter=100000, atol=1e-05, debug=False, silent=False):
    """
    Relaxes a structure of atoms, according to a Lennard-Jones potential
    :param structure: A 3-dimensional numpy array of atom-positions
    :param r_min: The distance where two atoms are at the lowest energy
    :param epsilon: The negative energy at r_min
    :param max_iter: Max number of iterations in the loop
    :param atol: Absolute tolerance in the loop
    :param debug:
    :param silent:
    :return: A relaxed numpy structure with same shape as the starting structure
    """
    def linesearch(struct, grad, r_m, eps, min_gamma=0, max_gamma=.5, num_gammas=3, num_iter=15):
        """
        Linesearch that looks for the step-size that lowers the energy as much as possible
        :param struct:
        :param grad:
        :param r_m:
        :param eps:
        :param min_gamma:
        :param max_gamma:
        :param num_gammas:
        :param num_iter:
        :return:
        """
        gms = []
        engs = []
        for num_iters in range(num_iter):
            engs = []
            gms = np.linspace(min_gamma, max_gamma, 101)
            for gm in gms:
                struct_test = struct - gm * grad
                engs.append(lennard_jones(struct_test, r_min=r_m, epsilon=eps))

            gamma_opts = np.sort(gms[np.argsort(engs)][:num_gammas])
            min_gamma, max_gamma = gamma_opts[[0, -1]]

        gm = gms[np.argmin(engs)]
        if gm == 0:
            gm = 0.001
        return gm

    structure_test = structure.copy()
    energies = [lennard_jones(structure_test, r_min, epsilon)]

    k = 10
    # TODO: Make k more general

    is_active = True
    while is_active is True:
        gradient = gradient_lennard_jones(structure_test, r_min, epsilon)

        gamma = linesearch(structure_test, gradient, r_min, epsilon)
        gamma = gamma * ((1 + np.min(abs(gradient))) / (1 + np.max(abs(gradient))))
        # TODO: This gamma shouldn't be here, but is necessary so the structure doesn't explode
        displacement = -gamma * gradient
        structure_test = structure + displacement

        energies.append(lennard_jones(structure_test, r_min, epsilon))

        if energies[-1] < energies[-2]:
            structure = structure_test.copy()

        if len(energies) > k and np.all(np.isclose(np.diff(np.array(energies[-k:])), 0, atol=atol)):
            if silent is False:
                print("Convergence in differences")
            is_active = False
        elif len(energies) > max_iter:
            if silent is False:
                print("Max size reached..")
            is_active = False

    if debug is True:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(energies)), energies)
        plt.grid()
        plt.xlim(0, len(energies) - 1)
        plt.ylim(min(energies) - 1, min([max(energies), 100]) + 1)
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.show()

    return structure


def gradient_features(structure, features, centroids, lam, eta, r_center, r_cutoff, upscale=1):
    """
    Calculates the gradient of a structure with features rho^I and rho^II.
    Should be used in the future
    :param structure:
    :param features:
    :param centroids:
    :param lam:
    :param eta:
    :param r_center:
    :param r_cutoff:
    :param upscale:
    :return:
    """
    def gradient_rho1(coordinates, r, lamd, eye, cos_cut, sin_cut, up_scale):
        coordinates_shortened = remove_diagonal(coordinates)
        r_shortened = remove_diagonal(r)

        grad_rho11 = np.zeros(r.shape)
        grad_rho12 = np.zeros(r.shape)
        if np.any(r[~eye] == 0):
            print(r)
            raise ValueError("Division by zero...")

        grad_rho11[~eye] = cos_cut[~eye] * np.exp(-r[~eye]/lamd) / r[~eye] * coordinates[~eye]
        grad_rho11[eye] = np.sum(cos_cut[~eye].reshape(r_shortened.shape) * np.exp(
            -r_shortened/lamd) / r_shortened * coordinates_shortened, axis=1)
        grad_rho11 = grad_rho11 / lamd

        grad_rho12[~eye] = sin_cut[~eye] * np.exp(-r[~eye]/lamd) / r[~eye] * coordinates[~eye]
        grad_rho12[eye] = np.sum(
            sin_cut[~eye].reshape(r_shortened.shape) * np.exp(-r_shortened/lamd) / r_shortened * coordinates_shortened,
            axis=1)

        gradient_1 = grad_rho11 + grad_rho12

        return up_scale * gradient_1

    def gradient_rho2(coordinates, r, eta2, r_centrum, r_c, eye, cos_cut, sin_cut):
        xyz_ij_shortened = remove_diagonal(coordinates)
        r_ij_shortened = remove_diagonal(r)

        grad_rho21 = np.zeros(r.shape)
        grad_rho22 = np.zeros(r.shape)

        grad_rho21[~eye] = cos_cut[~eye] * np.exp(-eta2*((r[~eye]-r_centrum)/r_c)**2) * (
                1 - r_centrum / r[~eye]) * coordinates[~eye]
        grad_rho21[eye] = np.sum(
            cos_cut[~eye].reshape(r_ij_shortened.shape) * np.exp(-eta2*((r_ij_shortened-r_centrum)/r_c)**2) * (
                    1 - r_centrum / r_ij_shortened) * xyz_ij_shortened, axis=1)
        grad_rho21 = grad_rho21 * 2*eta2/r_c**2

        grad_rho22[~eye] = sin_cut[~eye] * np.exp(-eta2*((r[~eye]-r_centrum)/r_c)**2) / r[~eye] * coordinates[~eye]
        grad_rho22[eye] = np.sum(sin_cut[~eye].reshape(r_ij_shortened.shape) * np.exp(
            -eta2*((r_ij_shortened-r_centrum)/r_c)**2) / r_ij_shortened * xyz_ij_shortened, axis=1)
        gradient_2 = grad_rho21 + grad_rho22

        return gradient_2

    natoms, ndims = structure.shape
    r_ij = compute_euclidian_distances(structure)

    cutoff = cutoff_function(r_ij, r_cutoff)
    grad_cutoff = gradient_cutoff(r_ij, r_cutoff)
    eye_index = np.eye(r_ij.shape[0], dtype=bool)

    cluster_list, cluster_index = cluster_distance(features, centroids, option='list')
    alpha1 = features[:, 0] - centroids[cluster_index, 0]
    alpha2 = features[:, 1] - centroids[cluster_index, 1]

    if np.any(cluster_list == 0):
        alpha1[cluster_list == 0] = 1
        alpha2[cluster_list == 0] = 1
        cluster_list[cluster_list == 0] = 1
    else:
        alpha1 = alpha1 / cluster_list
        alpha2 = alpha2 / cluster_list

    grad_ds = np.zeros(structure.shape)
    for i in range(ndims):
        dim = structure[:, i].reshape(natoms, 1)
        dim = dim.T - dim

        grho1 = gradient_rho1(dim, r_ij, lam, eye_index, cutoff, grad_cutoff, upscale)
        grho2 = gradient_rho2(dim, r_ij, eta, r_center, r_cutoff, eye_index, cutoff, grad_cutoff)

        grad_ds[:, i] = np.sum(grho1 * alpha1 + grho2 * alpha2, axis=1)

    return grad_ds


def minimize_cluster_distance(
        structure, centroids, lam, r_cutoff, r_center, eta, upscale=1, max_iter=100000, atol=1e-05,
        debug=False, silent=False
):
    """
    Minimizes the total cluster distance of a structure, with respect to a number of centroids.
    :param structure:
    :param centroids:
    :param lam:
    :param r_cutoff:
    :param r_center:
    :param eta:
    :param upscale:
    :param max_iter:
    :param atol:
    :param debug:
    :param silent:
    :return: An optimized structure, according to the total cluster distance
    """
    def linesearch(struct, grad, centr, lambd, r_cut, r_cent, et, min_gamma=0, max_gamma=1, num_gammas=6, num_iter=5):
        """

        :param struct:
        :param grad:
        :param centr:
        :param lambd:
        :param r_cut:
        :param r_cent:
        :param et:
        :param min_gamma:
        :param max_gamma:
        :param num_gammas:
        :param num_iter:
        :return:
        """
        gms = []
        dists = []
        for iteration in range(num_iter):
            dists = []
            gms = np.linspace(min_gamma, max_gamma, num_gammas)
            for gm in gms:
                struct_test = struct - gm * grad
                feature = calculate_features(struct_test, lambd, et, r_cent, r_cut)
                dists.append(cluster_distance(feature, centr))

            gamma_opts = np.sort(gms[np.argsort(dists)][:num_gammas])
            min_gamma, max_gamma = gamma_opts[[0, -1]]

        gm = gms[np.argmin(dists)]
        if gm == 0:
            gm = 0.001
        return gm

    features = calculate_features(structure, lam, eta, r_center, r_cutoff, upscale)
    distances = [cluster_distance(features, centroids)]

    means = [np.mean(distances)]

    n = 10
    # TODO: Make n more general

    is_active = True
    while is_active is True:
        features = calculate_features(structure, lam, eta, r_center, r_cutoff, upscale)

        gradient = gradient_features(structure, features, centroids, lam, eta, r_center, r_cutoff, upscale)
        gamma = linesearch(structure, gradient, centroids, lam, r_cutoff, r_center, eta)

        displacement = -gamma * gradient
        structure = structure + displacement

        distances.append(cluster_distance(features, centroids))
        means.append(np.sum(np.mean(distances[-n*10:])))

        if len(distances) > n and np.all(np.isclose(np.diff(np.array(distances[-n:])), 0, atol=atol)):
            if silent is False:
                print("Convergence of the differences")
            is_active = False
        elif len(means) > n and np.all(np.isclose(np.diff(np.array(means[-n:])), 0, atol=atol/10)):
            if silent is False:
                print("Convergence of the mean")
            is_active = False
        elif np.isclose(np.sum(abs(gradient)), 0):
            if silent is False:
                print("Convergence of the gradient")
            is_active = False
        elif len(distances) > max_iter:
            if silent is False:
                print("Max size reached..")
            is_active = False

    if debug is True:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(distances)), distances, label='Cluster distances')
        plt.plot(range(len(means)), means, label='Cumulative mean')
        plt.legend()
        plt.grid()
        plt.xlim(0, len(distances)-1)
        plt.ylim(min(distances)-1, max(distances)+1)
        plt.xlabel('Iteration')
        plt.ylabel('Total cluster distance')
        plt.show()

    return structure


def n_min(arr, n):
    """
    This function gives indices for the n lowest distances.
    :param arr: Numpy array of positions
    :param n: Number of indices.
    :return: Two numpy arrays, which includes the row- and column-indices, sorted so arr(row_indices[0], col_indices[0])
    gives the lowest distance found in the structure.
    """
    flat_indices = np.argpartition(arr.ravel(), n - 1)[:n]
    row_indices, col_indices = np.unravel_index(flat_indices, arr.shape)
    min_elements = arr[row_indices, col_indices]
    min_elements_order = np.argsort(min_elements)
    row_indices, col_indices = row_indices[min_elements_order], col_indices[min_elements_order]
    return row_indices, col_indices


def pair_sorting(row_main, row_test):
    """
    This sorts a second array of positions, according to the first, so the n'th entry in the resulting row, pairs the
    n'th entry in the main row.
    :param row_main: Numpy array of positions.
    :param row_test: Numpy array of positions, being sorted according to row_main.
    :return: Two numpy arrays, where the second is the sorted array.
    """
    num_xyz, num_dim = row_main.shape

    if num_xyz != row_test.shape[0] or num_dim != row_test.shape[1]:
        raise ValueError("row1 and row2 should have the same number of dimensions!")

    dists = np.zeros([num_xyz, num_xyz])
    for f in range(num_dim):
        row1f = np.repeat(row_main[:, f], num_xyz).reshape(num_xyz, num_xyz)
        row2f = np.repeat(row_test[:, f], num_xyz).reshape(num_xyz, num_xyz)
        dists = dists + (row1f - row2f.T) ** 2

    dists = np.sqrt(dists)

    row_result = np.zeros([num_xyz, 2])
    for n in range(num_xyz):
        x, y = np.where(dists == dists.min())
        row_result[x, :] = row_test[y, :]
        dists[x, :] = 10 * np.max(dists)
        dists[:, y] = 10 * np.max(dists)

    return row_main, row_result


def generate_structure(natoms, scale=5, sigma=0, ndims=3):
    structure = scale * np.random.normal(sigma, 1, size=(natoms, ndims))
    structure = structure % scale
    return structure


def rotate_structure(structure, nrotations=6):
    structure = structure.astype(float)

    num_atoms = structure.shape[0]
    angle = 2 * math.pi / nrotations
    rot_matrix = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])

    struct = np.zeros([nrotations * num_atoms, 2])
    rotated = structure.copy()
    for n in range(nrotations):
        struct[n*num_atoms:(n+1)*num_atoms, :] = rotated.copy()
        for i in range(num_atoms):
            rotated[i, :] = np.dot(rot_matrix, rotated[i, :])

    test = np.all(np.isclose(struct, 0), axis=1)
    test[0] = False
    struct = struct[~test, :]

    return struct


def generate_lj_minimum(r_min=1):
    x = np.arange(0, 3)*r_min
    y = np.zeros(x.shape)
    s = rotate_structure(np.append(x[:2], y[:2]).reshape(2, 2).T)
    x = np.append(x, np.mean(x[-2:]))
    y = np.append(y, s[3, 1])
    return rotate_structure(np.append(x, y).reshape(2, len(x)).T)
