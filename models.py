import numpy as np
import warnings
import random
from maths import split_seq, sigmoid, lennard_jones, cluster_distance, calculate_features
from maths import pair_sorting, relax_structure, minimize_cluster_distance, compute_sizes
from plot_function import plot_structure, plot_clustering
warnings.filterwarnings('error')


class KMeans:
    """
    KMeans is a model that clusters data into a number of specified clusters, via an averaging function.
    This is an unsupervised machine-learning model, which means the classes given from the clustering, is only relative
    to the other clusters.
    """
    def __init__(self, features, num_clusters, num_repeats=10, normalizing=True, silent=False):
        self.silent = silent
        if self.silent is False:
            print("Initializing model")
        if normalizing is True and np.all(np.max(features, axis=0) != 0):
            self.norm_factors = 1/(np.max(features, axis=0)-np.min(features, axis=0))
        else:
            self.norm_factors = np.ones((1, features.shape[1]))
        self.features = features*self.norm_factors
        self.num_clusters = num_clusters
        self.num_dimensions = self.features.shape[1]
        self.num_feature_vectors = self.features.shape[0]

        self.old_clusters = np.array([])
        self.clusters = np.array([])
        self.opt_clusters = np.array([])

        centroid_initialize = random.sample(range(0, self.num_feature_vectors), self.num_clusters)
        self.norm_centroids = self.features[centroid_initialize, :]
        self.opt_centroids = np.array([])

        self.dist = []
        self.opt_dist = np.array([])
        self.repeats_left = num_repeats
        self.returning = {}

    def run_model(self):
        if self.silent is False:
            print("Running...")

        while self.repeats_left > 0:
            KMeans.update_clusters(self)
            self.dist = KMeans._centroid_feature_dist(self)

            KMeans.loop_run(self)
            KMeans.optimize_model(self)

            self.repeats_left = self.repeats_left - 1
            KMeans.reset_model(self)
            if self.silent is False:
                print("Runs left: {}".format(self.repeats_left))

        if self.silent is False:
            print("Done!\n")
        return self.opt_centroids, self.opt_clusters

    def update_clusters(self):
        self.old_clusters = self.clusters.copy()

        dist_matrix = np.array([])
        for k in range(0, self.num_clusters):
            dists = np.sum((self.norm_centroids[k, :] - self.features) ** 2, axis=1) ** (1 / 2)
            dist_matrix = np.append(dist_matrix, dists)

        dist_matrix = np.reshape(dist_matrix, (self.num_clusters, self.num_feature_vectors)).T
        self.clusters = np.argmin(dist_matrix, axis=1)

    def update_centroids(self):
        for k in range(0, self.num_clusters):
            try:
                self.norm_centroids[k, :] = np.mean(self.features[np.where(self.clusters == k), :], axis=1)
            except RuntimeWarning:
                self.norm_centroids[k, :] = np.zeros(self.num_dimensions)
                if self.silent is False:
                    print("A cluster with 0 data-points was found! \n\tCorrecting...")

    def convergence_test(self):
        if self.old_clusters.shape == self.clusters.shape:
            has_converged = (self.old_clusters == self.clusters).all()
        else:
            has_converged = False
        return has_converged

    def loop_run(self):
        while not KMeans.convergence_test(self):
            KMeans.update_centroids(self)
            KMeans.update_clusters(self)

    def reset_model(self):
        self.old_clusters = np.array([])
        self.clusters = np.array([])

        centroid_initialize = random.sample(range(0, self.num_feature_vectors), self.num_clusters)
        self.norm_centroids = self.features[centroid_initialize, :]

    def optimize_model(self):
        if self.opt_dist.size == 0 or KMeans._centroid_feature_dist(self) < self.opt_dist:
            self.opt_clusters = self.clusters.copy()
            self.opt_centroids = self.norm_centroids.copy() / self.norm_factors
            self.opt_dist = KMeans._centroid_feature_dist(self)

    def _centroid_feature_dist(self):
        dist_matrix = self.norm_centroids[self.clusters, :] - self.features
        dist_vector = np.sum(dist_matrix ** 2, axis=1) ** (1 / 2)
        return np.sum(dist_vector)

    def add_feature(self, feature):
        try:
            if feature.ndim == 1:
                feature = np.array([feature])
            num_features = feature.shape[0]
            dist_matrix = np.array([])
            for k in range(0, self.num_clusters):
                dists = np.sum((self.norm_centroids[k, :] - feature) ** 2, axis=1) ** (1 / 2)
                dist_matrix = np.append(dist_matrix, dists)
            dist_matrix = np.reshape(dist_matrix, (self.num_clusters, num_features)).T
            local_clusters = np.argmin(dist_matrix, axis=1)
            return local_clusters
        except TypeError:
            print("Please use a 1- or 2-dimensional numpy array")


class LinearRegression:
    """
    Methods give_regression() and predict() can be used. This gives linear regression from the given matrices and
    solutions to the equation f*X=y.
    give_regression() gives the best approximations for X, which lowers the errors as much as possible.
    predict() gives the opportunity to predict the y, using the approximated X.

    If ridge regression is preferred, the regularization constant lambda(here lam) can be defined. If Ridge=True is set,
    a default value of lam=0.01 is used.
    """
    def __init__(self, features, solution, lam=0, ridge=False):
        if features.shape[0] > features.shape[1]:
            self.features = features
        else:
            self.features = features.T

        if lam != 0:
            self.lam = lam
        elif ridge is True:
            self.lam = 0.01
        else:
            self.lam = 0

        self.solution = solution
        self.coefs = np.array([])
        self.prediction = np.array([])

    def give_regression(self):
        a = np.matmul(self.features.T, self.features)
        b = np.linalg.inv(a + self.lam*np.eye(len(a)))
        c = np.matmul(b, self.features.T)
        self.coefs = np.dot(c, self.solution)
        return self.coefs

    def predict(self, feature):
        energies = np.dot(feature, self.coefs)
        self.prediction = energies
        return self.prediction

    def give_error(self, solution):
        mae = np.mean(np.abs(self.prediction-solution))
        return mae


def cross_validation(features, solution, k, lam=0, ridge=False):
    """
    k-fold Cross Validation splits the data into k folds, before using regression to predict energies.
    As the different folds will give different Mean Absolute Error, the fold that decreases MAE as much as possible
    is returned, in the form of coefficients used in regression.
    """
    if lam != 0:
        lam = lam
    elif ridge is True:
        lam = 0.01
    else:
        lam = 0

    min_mae = np.array([])
    opt_regression = np.array([])

    split = split_seq(range(len(solution)), k)

    for i in range(len(split)):
        test_features = features[split[i], :]
        test_solution = solution[split[i]]
        testing_features = np.delete(features, split[i], axis=0)
        testing_solution = np.delete(solution, split[i], axis=0)

        linear = LinearRegression(test_features, test_solution, lam=lam)
        regression = linear.give_regression()
        linear.predict(testing_features)
        mae = linear.give_error(testing_solution)

        if min_mae.size > 0:
            if mae < min_mae:
                min_mae = mae
                opt_regression = regression
        else:
            min_mae = mae
            opt_regression = regression

    return opt_regression, min_mae


class RolemodelLearning:
    """
    A method used for the search for a global minimum of a structure.
    This is done by shifting between relaxing the structure according to a potential, and minimizing the total cluster
    distance.
    The number of rolemodels determine how many of the centroids are used in the minimizing of the cluster distance.
    """
    def __init__(
            self, structure, num_centroids, num_rolemodels, niter, parameters,
            max_iter=1000000, atol=1e-05, silent=False, debug=False
    ):
        self.structure = structure
        self.num_centroids = num_centroids
        self.num_rolemodels = num_rolemodels
        self.niter = niter

        self.parameters = parameters.copy()
        self.r_min = self.parameters.pop('r_min')
        self.epsilon = self.parameters.pop('epsilon')
        self.eta = self.parameters.pop('eta')
        self.lam = self.parameters.pop('lam')
        self.r_center = self.parameters.pop('r_center')
        self.r_cutoff = self.parameters.pop('r_cutoff')
        if 'upscale' in self.parameters:
            self.upscale = self.parameters.pop('upscale')
        else:
            self.upscale = 1

        if 'rattle_factor' in self.parameters:
            if self.parameters['rattle_factor'] < 0 or self.parameters['rattle_factor'] > 1:
                raise ValueError("rattle_factor should between 0 and 1.")
            else:
                self.rattle_factor = self.parameters.pop('rattle_factor')
        else:
            self.rattle_factor = 0

        if 'global_minimum' in self.parameters:
            self.global_minimum = self.parameters.pop('global_minimum')
            self.global_min_iter = 0
        else:
            self.global_minimum = None
            self.global_min_iter = None

        if self.parameters:
            raise ValueError("Non-used parameters given")

        self.max_iter = max_iter
        self.atol = atol
        self.silent = silent
        self.debug = debug

        self.grades = np.zeros(self.num_centroids)
        self.grade_results = self.grades[np.newaxis].copy()

        self.energy_start = lennard_jones(self.structure, self.r_min)
        self.energies = [self.energy_start]
        self.energy_new = None
        self.energy_opt = self.energy_start

        self.features = calculate_features(self.structure, self.lam, self.eta, self.r_center, self.r_cutoff)
        self.centroids, self.clusters = KMeans(self.features, self.num_centroids, self.num_rolemodels, silent=True
                                               ).run_model()

        self.cluster_dist_start = cluster_distance(self.features, self.centroids)
        self.cluster_distances = [self.cluster_dist_start]
        self.cluster_dist_new = None
        self.cluster_dist_opt = self.cluster_dist_start

        self.structure_relaxed = self.structure.copy()
        self.structure_opt = self.structure_relaxed.copy()

        self.rolemodel_index = list(range(self.num_centroids))
        self.rolemodels = self.centroids[self.rolemodel_index, :]

    def run_model(self):
        if self.silent is False:
            print("Starting energy: {:.3e}".format(self.energy_start))
            print("Starting cluster distance: {:.3e}".format(self.cluster_dist_start))
            print("Running model ... ")

        RolemodelLearning.relax_structure(self)

        rolemodel_index = RolemodelLearning.choose_rolemodels(self)
        RolemodelLearning.plot_struct_clustering(self, rolemodel_index)

        for iteration in range(1, self.niter + 1):
            if self.silent is False:
                print("\tIteration {}".format(iteration))

            RolemodelLearning.rattle_structure(self)
            RolemodelLearning.minimize_cluster_distance(self)

            rolemodel_index = RolemodelLearning.choose_rolemodels(self)
            RolemodelLearning.plot_struct_clustering(self, rolemodel_index)

            RolemodelLearning.relax_structure(self)

            self.grades[rolemodel_index] += 2*sigmoid(self.energies[-3] - self.energies[-1]) - 1
            self.grade_results = np.append(self.grade_results, self.grades[np.newaxis], axis=0)
            rolemodel_index = RolemodelLearning.choose_rolemodels(self)
            RolemodelLearning.plot_struct_clustering(self, rolemodel_index)

            if self.global_minimum is not None and np.isclose(self.energies[-1], self.global_minimum, atol=self.atol):
                self.global_min_iter = iteration
                break

        if self.global_min_iter == 0:
            self.global_min_iter = 1.1 * self.niter

        if self.silent is False:
            print("Done!")
            if self.global_min_iter is not None:
                if self.global_min_iter < self.niter + 1:
                    print("\nThe global minimum was found at iteration number {}!".format(self.global_min_iter))
                else:
                    print("\nThe global minimum was not reached..")
            else:
                print("\nThe minimum energy is: {:.3e}".format(self.energy_opt))
                print("A change in energy is found to be: {:.3e}".format(self.energy_opt - self.energy_start))
                print("\nThe cluster distance is: {:.3e}".format(self.cluster_dist_opt))
                print("A change in cluster distance is found to be: {:.3e}".format(
                    self.cluster_dist_opt - self.cluster_dist_start)
                )

    def rattle_structure(self):
        displacement = np.random.normal(size=self.structure_relaxed.shape)
        displacement = displacement / np.max(abs(displacement)) * self.rattle_factor * self.r_min
        self.structure_relaxed = self.structure_relaxed + displacement

    def minimize_cluster_distance(self):
        if self.silent is False:
            print("\nMinimizing cluster-distance ...")

        self.structure_relaxed = minimize_cluster_distance(
            self.structure_relaxed, self.rolemodels, lam=self.lam, r_cutoff=self.r_cutoff, r_center=self.r_center,
            eta=self.eta, upscale=self.upscale, max_iter=self.max_iter, atol=self.atol, debug=False,
            silent=True
        )
        self.features = calculate_features(
            self.structure_relaxed, self.lam, self.eta, self.r_center, self.r_cutoff, upscale=self.upscale
        )
        RolemodelLearning.add_energy_cluster_dist(self)

    def relax_structure(self):
        if self.silent is False:
            print("\nMinimizing energy ...")

        self.structure_relaxed = relax_structure(
            self.structure_relaxed, r_min=self.r_min, epsilon=self.epsilon,
            max_iter=self.max_iter, atol=self.atol, debug=False, silent=True
        )
        self.features = calculate_features(
            self.structure_relaxed, self.lam, self.eta, self.r_center, self.r_cutoff, upscale=self.upscale
        )
        RolemodelLearning.add_energy_cluster_dist(self)

    def choose_rolemodels(self):
        centroids, self.clusters = KMeans(
            self.features, self.num_centroids, self.num_rolemodels, silent=True
        ).run_model()
        _, self.centroids = pair_sorting(self.centroids, centroids)

        random.shuffle(self.rolemodel_index)

        centroid_sizes = compute_sizes(self.centroids).flatten()
        not_zero = centroid_sizes > 0.1
        num_not_zero = np.sum(not_zero)

        num_rolemodels = max([min([num_not_zero, self.num_rolemodels]), 1])

        rolemodel_index = np.array(self.rolemodel_index)[not_zero[self.rolemodel_index]]
        if len(rolemodel_index) == 0:
            rolemodel_index = self.rolemodel_index.copy()
        else:
            rolemodel_index = rolemodel_index[:num_rolemodels]

        self.rolemodels = self.centroids[rolemodel_index, :]
        if self.rolemodels.shape[0] == 0:
            raise ValueError("0 rolemodels equipped!")

        for rm in range(self.rolemodels.shape[0]):
            size = np.sqrt(np.sum(self.rolemodels[rm, :]**2))
            if size < 0.1:
                RolemodelLearning.plot_struct_clustering(self, self.rolemodel_index)
        return rolemodel_index

    def add_energy_cluster_dist(self):
        energy = lennard_jones(self.structure_relaxed, r_min=self.r_min)
        distance = cluster_distance(self.features, self.rolemodels)

        self.energies.append(energy)
        self.cluster_distances.append(distance)

        if distance < self.cluster_dist_opt:
            self.cluster_dist_opt = distance

        if energy < self.energy_opt:
            self.energy_opt = energy
            self.structure_opt = self.structure_relaxed.copy()

    def plot_struct_clustering(self, rolemodel_index):
        if self.debug is True:
            plot_structure(
                self.structure_relaxed, title='Energy: {:.2e}'.format(self.energies[-1])
            )
            plot_clustering(
                self.features, self.centroids, clusters=self.clusters,
                figsize=(7, 7), alpha=1, fs=50, centroid_emphasis=rolemodel_index
            )

    @property
    def give_structure(self):
        return self.structure_opt

    @property
    def give_energies(self):
        return self.energies

    @property
    def give_energy(self):
        return self.energy_opt

    @property
    def give_distances(self):
        return self.cluster_distances

    @property
    def give_distance(self):
        return self.cluster_dist_opt

    @property
    def give_grades(self):
        return self.grades

    @property
    def give_grade_results(self):
        return self.grade_results

    @property
    def give_iteration(self):
        if self.global_min_iter is not None:
            return self.global_min_iter
