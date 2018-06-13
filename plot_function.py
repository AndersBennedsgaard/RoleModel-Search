import numpy as np
from maths import compute_sizes, cluster_distance
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.cm as cm


def plot_clustering(
        features, centroids, title=None, xlabel=r"$\rho_i^{I}$", ylabel=r"$\rho_i^{II}$",
        alpha=0.3, cmarker='*', cs=50, clabel='Centroids', calpha=0.7, fs=15,
        figsize=(7, 7), clusters=None, savefig=False, figtitle='clustering.png', linewidth='.3', edgecolor='black',
        centroid_emphasis=None, emarker='x', es=70, ecolor='red', elabel='Role-models', legend=True
):
    nclusters = centroids.shape[0]
    if clusters is None:
        cluster_dist = np.zeros([features.shape[0], nclusters])
        for i in range(nclusters):
            cluster_dist[:, i] = compute_sizes(features - centroids[i, :]).flatten()

        clusters = np.argmin(cluster_dist, axis=1)

    if centroid_emphasis is None:
        centroid_emphasis = []

    if title is None:
        if len(centroid_emphasis) != 0:
            cluster_dist = cluster_distance(features, centroids[centroid_emphasis, :])
            title = 'Total role-model cluster distance: {:.2f}'.format(cluster_dist)
        else:
            cluster_dist = cluster_distance(features, centroids)
            title = 'Total cluster distance: {:.2f}'.format(cluster_dist)

    colors = cm.rainbow(np.linspace(0, 1, nclusters))
    plt.figure(figsize=figsize)
    for cluster in np.sort(np.unique(clusters)):
        index = np.where(clusters == cluster)

        plt.scatter(
            features[index, 0], features[index, 1],
            color=colors[cluster], label='Cluster '+str(cluster), alpha=alpha, marker='.', s=fs, edgecolor=edgecolor,
            linewidth=linewidth
        )

    if calpha == 0:
        clabel = ''

    cluster_index = ~np.isin(range(nclusters), centroid_emphasis)
    plt.scatter(
        centroids[cluster_index, 0], centroids[cluster_index, 1], marker=cmarker, c='blue', alpha=calpha,
        edgecolor='black', s=cs, label=clabel
    )
    if len(centroid_emphasis) != 0:
        plt.scatter(
            centroids[~cluster_index, 0], centroids[~cluster_index, 1], marker=emarker, c=ecolor, alpha=1,
            edgecolor='black', s=es, label=elabel
        )

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16)
    if legend is True:
        plt.legend(fontsize=12)
    plt.grid()
    if savefig is True:
        plt.savefig(figtitle)
    plt.show()


def plot_structure(
        structure, title='', xlabel='x', ylabel='y', figsize=(7, 7), s=400, cmap='copper', edgecolor='black',
        savefig=False, figtitle='structure.png', structure2=None, label1="Structure 1", label2="Structure 2"
):
    natoms, ndims = structure.shape
    plt.figure(figsize=figsize)
    if ndims == 3:
        structure = structure[np.argsort(structure[:, 2]), :]
        plt.scatter(structure[:, 0], structure[:, 1], s=s, c=structure[:, 2], cmap=cmap, edgecolor=edgecolor)
        plt.colorbar()
    elif ndims == 2:
        if structure2 is not None:
            plt.scatter(structure[:, 0], structure[:, 1], s=s, cmap=cmap, edgecolor=edgecolor, label=label1)
            plt.scatter(structure2[:, 0], structure2[:, 1], s=s, cmap='Blues', edgecolors=edgecolor, label=label2)
        else:
            plt.scatter(structure[:, 0], structure[:, 1], s=s, cmap=cmap, edgecolor=edgecolor)
    else:
        raise ValueError("structure-array should be 2- or 3-dimensional")
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title, fontsize=20)
    if structure2 is not None:
        plt.legend(loc='best', fontsize=14)
    plt.axis('equal')
    plt.grid()
    if savefig is True:
        plt.savefig(figtitle)
    plt.show()


def hist_ranges(
        distances1, distances2, num_bins=40, bins=None, title='Histogram of all internal bond-lengths',
        xlabel='Distances', ylabel='Number of instances', figsize=(10, 7), c1='red', c2='blue', edgecolor1='black',
        edgecolor2='black', label1='Structure 1', label2='Structure 2', alpha=0.5
):
    bins = bins
    num_bins = num_bins

    if bins is None:
        bins = np.linspace(0, np.max(np.append(distances1, distances2)), num_bins+1)

    plt.figure(figsize=figsize)
    plt.hist(distances1.flatten(), bins=bins, color=c1, alpha=alpha, edgecolor=edgecolor1, label=label1)
    plt.hist(distances2.flatten(), bins=bins, color=c2, alpha=alpha, edgecolor=edgecolor2, label=label2)
    plt.legend(loc='best')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlim(0,bins.max())
    plt.title(title, fontsize=14)
    plt.show()


def subplots_horizontal(
        energies, distances, title, figsize=(10, 5), first_ylabel=r'$y_1$', second_ylabel=r'$y_2$', xlabel='x'
):
    plt.figure(figsize=figsize)
    plt.subplot(2, 1, 1)
    plt.plot(energies)
    plt.xlim(0, len(energies) - 1)
    plt.ylim(min(energies) - 1, min(max(energies), 100) + 1)
    plt.ylabel(first_ylabel)
    plt.title(title)
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(distances)
    plt.xlim(0, len(distances) - 1)
    plt.ylim(min(distances) - 1, max(distances) + 1)
    plt.xlabel(xlabel)
    plt.ylabel(second_ylabel)
    plt.grid()
    plt.show()


def density_error(x, y, y_error, figsize=(10, 5), xlabel='', ylabel='', alpha=0.2, edgecolor='none'):
    if y_error.size == y.size:
        y_error = np.append(y_error, y_error, axis=0).reshape(2, y.size).T
    elif y_error.size != y.size * 2:
        raise ValueError("y_error should have the same or double the size of y")

    plt.figure(figsize=figsize)
    for n in range(y.shape[1]):
        plt.plot(x, y[:, n])
        plt.fill_between(x, y[n] - y_error[:, 0], y[n] + y_error[:, 1], alpha=alpha, edgecolor=edgecolor)
    plt.xlim(np.min(x), x[-1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def energy_distance_correlation(
        distances, energies, title='', xlabel='', ylabel='', figsize=(10, 5), color='black', fontsize=12
):
    plt.figure(figsize=figsize)
    plt.plot(distances, energies)
    for n in range(len(energies)):
        pl.text(distances[n], energies[n], n, color=color, fontsize=fontsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.show()
