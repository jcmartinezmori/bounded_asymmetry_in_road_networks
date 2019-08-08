import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
from itertools import combinations
from scipy.optimize import curve_fit
import pickle
import powerlaw
import gc

__author__ = 'jm2638@cornell.edu'


def main():

    # place tuple (name [string], (north_lat [float], south_lat [float], east_lon [float], west_lon [float]))
    # for each place under study (for figures 2 and 3 (and supplementary material))
    places = [
        ('Buenos Aires, Argentina', (-34.5583, -34.6519, -58.4819, -58.3504)),
        ('Athens, Greece', (38.0125, 37.9507, 23.6932, 23.7653)),
        ('Quito, Ecuador', (-0.1428, -0.2547, -78.5368, -78.4297)),
        ('Chennai, India', (13.1263, 13.0370, 80.2201, 80.3015)),
        ('Vancouver, BC', (49.3024, 49.2147, -123.2666, -123.0283)),
        ('Tokyo, Japan', (35.7047, 35.6609, 139.7277, 139.8000)),
        ('New Orleans, LA', (30.0379, 29.9097, -90.1600, -90.0422)),
        ('Manhattan, NYC', None),
        ('Barcelona, Spain', (41.4185, 41.3281, 2.1104, 2.2343)),
        ('Moscow, Russia', (55.8183, 55.6950, 37.5032, 37.7452)),
        ('Auckland, New Zealand', (-36.8030, -36.9123, 174.6878, 174.8508)),
        ('Mogadishu, Somalia', (2.0660, 2.0107, 45.2995, 45.3699))
    ]

    # color code for each place under study (for figure 4)
    colors = [
        (38/255, 132/255, 89/255),
        (26/255, 162/255, 201/255),
        (26/255, 162/255, 201/255),
        (26/255, 162/255, 201/255),
        (26/255, 162/255, 201/255),
        (26/255, 162/255, 201/255),
        (234/255, 175/255, 65/255),
        (234/255, 175/255, 65/255),
        (234/255, 175/255, 65/255),
        (234/255, 175/255, 65/255),
        (234/255, 175/255, 65/255),
        (218/255, 34/255, 52/255)
    ]

    # line style for each place under study (for figure 4)
    linestyles = [
        (0, ()),
        (0, ()),
        (0, (1, 1)),
        (0, (5, 1)),
        (0, (3, 1, 1, 1)),
        (0, (3, 1, 1, 1, 1, 1)),
        (0, ()), (0, (1, 1)),
        (0, (5, 1)),
        (0, (3, 1, 1, 1)),
        (0, (3, 1, 1, 1, 1, 1)),
        (0, ())
    ]

    # plot figures 2 and 3 (and supplementary material)
    # ps.: be careful about memory overflow if running many cities at once. garbage collection is problematic.
    if True:
        for place in places:
            graph = load_graph(place)
            lengths = load_lengths(place, graph)
            del graph
            gc.collect()
            points = load_points(place, lengths)
            del lengths
            gc.collect()
            plot_scatter(place, points=points)  # this goes first because next function might mess things up
            plot_ccdf(place, points=points)
            del points
            gc.collect()

    # plot figure 4
    if False:
        plot_multiple_maxfactors(places, colors=colors, linestyles=linestyles)


def load_graph(place):
    """
    load graph networkx
    :param place: tuple
        (name [string], (north_lat [float], south_lat [float], east_lon [float], west_lon [float]))
    :return g: networkx graph
    """
    print('     Loading graph ...')
    name, bbox = place
    try:
        with open('./graphs_dir/{0}.pkl'.format(name), 'rb') as f:
            graph = pickle.load(f)
    except FileNotFoundError:
        if bbox is None:
            graph = ox.graph_from_place(name, network_type='drive')
        else:
            graph = ox.graph_from_bbox(bbox[0], bbox[1], bbox[3], bbox[2], network_type='drive')
        graph = max(nx.strongly_connected_component_subgraphs(graph), key=len)
        try:
            with open('./graphs_dir/{0}.pkl'.format(name), 'wb') as f:
                pickle.dump(graph, f, protocol=4)
        except MemoryError:
            print('         Warning: Dump unsuccessful ...')
    print('         Done!')
    return graph


def load_lengths(place, graph):
    """
    load lengths dictionary
    :param place: tuple
        (name [string], (north_lat [float], south_lat [float], east_lon [float], west_lon [float]))
    :param graph: networkx graph
    :return lengths: dictionary with
        key: node u id [int]
        value: dictionary with
            key: node v id [int]
            value: length of shortest path from u to v [float]
    """
    print('     Loading lengths ...')
    name, bbox = place
    try:
        with open('./lengths_dir/{0}.pkl'.format(name), 'rb') as f:
            lengths = pickle.load(f)
    except FileNotFoundError:
        lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight='length'))
        try:
            with open('./lengths_dir/{0}.pkl'.format(name), 'wb') as f:
                pickle.dump(lengths, f, protocol=4)
        except MemoryError:
            print('         Warning: Dump unsuccessful ...')
    print('         Done!')
    return lengths


def load_points(place, lengths):
    """
    load points list
    :param place: tuple
        (name [string], (north_lat [float], south_lat [float], east_lon [float], west_lon [float]))
    :param lengths: dictionary with
        key: node u id [int]
        value: dictionary with
            key: node v id [int]
            value: length of shortest path from u to v [float]
    :return points: list of tuples (length [float], asymmetry_factor [float])
    """
    print('     Loading points ...')
    name, bbox = place
    try:
        with open('./points_dir/{0}.pkl'.format(name), 'rb') as f:
            points = pickle.load(f)
    except FileNotFoundError:
        pairs = list(combinations(lengths.keys(), 2))
        points = []
        for pair in pairs:
            u, v = pair
            length = min(lengths[u][v], lengths[v][u])
            asymmetry_factor = max(lengths[u][v] / lengths[v][u], lengths[v][u] / lengths[u][v])
            points.append((length, asymmetry_factor))
        points.sort(key=lambda point: point[0])
        try:
            with open('./points_dir/{0}.pkl'.format(name), 'wb') as f:
                pickle.dump(points, f, protocol=4)
        except MemoryError:
            print('         Warning: Dump unsuccessful ...')
    print('         Done!')
    return points


def load_node_pairs(place, lengths=None):
    """
    load points list
    :param place: tuple
        (name [string], (north_lat [float], south_lat [float], east_lon [float], west_lon [float]))
    :param lengths: dictionary with
        key: node u id [int]
        value: dictionary with
            key: node v id [int]
            value: length of shortest path from u to v [float]
    :return points: list of tuples (length [float], asymmetry_factor [float])
    """
    print('     Loading node_pairs ...')
    name, bbox = place
    try:
        with open('./node_pairs_dir/{0}.pkl'.format(name), 'rb') as f:
            node_pairs = pickle.load(f)
    except FileNotFoundError:
        if lengths is None:
            lengths = load_lengths(place)
        pairs = list(combinations(lengths.keys(), 2))
        node_pairs = []
        for pair in pairs:
            u, v = pair
            length = min(lengths[u][v], lengths[v][u])
            asymmetry_factor = max(lengths[u][v] / lengths[v][u], lengths[v][u] / lengths[u][v])
            node_pairs.append(((u, v), asymmetry_factor, length))
        node_pairs.sort(key=lambda node_pair: node_pair[2])
        try:
            with open('./node_pairs_dir/{0}.pkl'.format(name), 'wb') as f:
                pickle.dump(node_pairs, f, protocol=4)
        except MemoryError:
            print('         Warning: Dump unsuccessful ...')
    print('         Done!')
    return node_pairs


def plot_scatter(place, points=None, max_asymmetry=None, max_length=None, num_bins=1000):
    """
    plot scatter of points along with their marginal frequencies
    :param place: tuple
        (name [string], (north_lat [float], south_lat [float], east_lon [float], west_lon [float]))
    :param points: list of tuples (length [float], asymmetry_factor [float])
    :param max_asymmetry: maximum asymmetry to include in plot
    :param max_length: maximum length to include in plot
    :param num_bins: number of bins for marginal frequencies
    :return:
    """
    print('     Plotting scatter ...')
    name, bbox = place
    if points is None:
        points = load_points(place)
    lengths, asymmetry_factors = zip(*points)

    plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(3, 3)
    ax_scatter = plt.subplot(gs[1:3, :2])
    ax_length_freq = plt.subplot(gs[0, :2], sharex=ax_scatter)
    ax_factor_freq = plt.subplot(gs[1:3, 2], sharey=ax_scatter)

    ax_scatter.scatter(lengths, asymmetry_factors, marker='.', s=1)
    ax_scatter.set_xlabel('Length [m]', fontsize=16)
    ax_scatter.set_ylabel('Asymmetry Factor', fontsize=16)
    if max_length is not None:
        ax_scatter.set_xlim(-100, max_length + 100)
    if max_asymmetry is not None:
        ax_scatter.set_ylim(-2.5, max_asymmetry + 2.5)
    ax_scatter.grid()

    ax_length_freq.hist(lengths, bins=num_bins, align='mid')
    ax_length_freq.set_ylabel('Frequency', fontsize=16)
    ax_length_freq.grid()

    ax_factor_freq.hist(asymmetry_factors, bins=num_bins, orientation='horizontal', align='mid')
    ax_factor_freq.set_xlabel('Frequency', fontsize=16)
    ax_factor_freq.set_xscale('log')
    ax_factor_freq.grid()

    plt.savefig('./figs_dir/scatter {0}.png'.format(name), format='png', dpi=200)
    print('         Done!')


def load_maxfactors(place, points):
    """
    load factor decay list
    :param place: tuple
        (name [string], (north_lat [float], south_lat [float], east_lon [float], west_lon [float]))
    :param points: list of tuples (length [float], asymmetry_factor [float])
    :return maxfactors: list of tupl es (min length [float], max asymmetry factor [float])
    """
    print('     Loading maxfactors ...')
    name, bbox = place
    try:
        with open('./maxfactors_dir/maxfactors {0}.pkl'.format(name), 'rb') as f:
            maxfactors = pickle.load(f)
    except FileNotFoundError:
        cpoints = points
        # cpoints = points.copy()  # careful with overwriting list vs. memory overflow
        lengths, asymmetry_factors = zip(*cpoints)
        max_asymmetry_factors = []
        min_lengths = np.arange(0, max(lengths), 10)
        for min_length in min_lengths:
            max_asymmetry_factor = 1
            above_min_length = []
            while cpoints:
                point = cpoints.pop()
                length, asymmetry_factor = point
                if length >= min_length:
                    above_min_length.append(point)
                    if asymmetry_factor > max_asymmetry_factor:
                        max_asymmetry_factor = asymmetry_factor
            max_asymmetry_factors.append(max_asymmetry_factor)
            cpoints = above_min_length
            if max_asymmetry_factor == 1:
                break
        max_asymmetry_factors.extend([1 for _ in range(len(min_lengths) - len(max_asymmetry_factors))])
        maxfactors = (np.array(min_lengths), np.array(max_asymmetry_factors))
        try:
            with open('./maxfactors_dir/maxfactors {0}.pkl'.format(name), 'wb') as f:
                pickle.dump(maxfactors, f, protocol=4)
        except MemoryError:
            print('         Warning: Dump unsuccessful ...')
    print('         Done!')
    return maxfactors


def plot_ccdf(place, points, thresholds=None):
    """
    plot ccdf
    :param place: tuple
        (name [string], (north_lat [float], south_lat [float], east_lon [float], west_lon [float]))
    :param points: list of tuples (length [float], asymmetry_factor [float])
    :param thresholds: list of floats of minimum length threshold to filter points and plot ccdf
    :return:
    """
    print('     Plotting ccdf ...')
    name, bbox = place
    cpoints = points
    # cpoints = points.copy()  # careful with overwriting list vs. memory overflow
    if thresholds is None:
        thresholds = [0, 250, 500, 1000, 1500, 3000, 4500]
    cmap = plt.get_cmap('Set1')
    colors = [cmap(i) for i in np.linspace(0, 1, len(thresholds))]
    plt.figure()
    for idx, threshold in enumerate(thresholds):
        above_threshold = []
        asymmetry_factors = []
        while cpoints:
            point = cpoints.pop()
            length, asymmetry_factor = point
            if length >= threshold:
                above_threshold.append(point)
                asymmetry_factors.append(asymmetry_factor)
        powerlaw.plot_ccdf(
            asymmetry_factors, color=colors[idx], linewidth=1.5, label='$Length \geq {0} \ m$'.format(threshold))
        cpoints = above_threshold
    plt.xlabel('Asymmetry Factor', fontsize=16)
    plt.ylabel('$P(X \geq x)$', fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig('./figs_dir/ccdf {0}.png'.format(name), format='png', dpi=200)
    print('         Done!')


def plot_multiple_maxfactors(places, colors, **kwargs):
    """
    plot multiple maxfactors (no fit)
    :param places: list of tuples
        (name [string], (north_lat [float], south_lat [float], east_lon [float], west_lon [float]))
    :return:
    """
    print('     Plotting multiple maxfactors ...')
    cmap = plt.get_cmap('Set1')
    plt.figure()
    for idx, place in enumerate(places):
        name, bbox = place
        maxfactors = load_maxfactors(place)
        min_lengths, max_asymmetry_factors = maxfactors
        if 'linestyles' in kwargs:
            plt.plot(min_lengths, max_asymmetry_factors, linestyle=kwargs['linestyles'][idx], color=colors[idx],
                     linewidth=.75, label='{0}'.format(name))
        else:
            plt.plot(min_lengths, max_asymmetry_factors, linestyle='-', color=colors[idx],
                     linewidth=.75, label='{0}'.format(name))
    plt.xlabel('Minimum Length [m]')
    plt.ylabel('Maximum Asymmetry Factor')
    plt.ylim(-5, 105)
    plt.xlim(-500, 10500)
    plt.grid()
    plt.legend()
    plt.savefig('./figs_dir/multiple maxfactors.png', format='png', dpi=400)
    print('         Done!')


if __name__ == '__main__':
    main()
