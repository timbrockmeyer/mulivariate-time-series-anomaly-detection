import torch
import networkx as nx
from sklearn.manifold import TSNE
from pyvis.network import Network
import matplotlib.pyplot as plt

def plot_embedding(edge_indices, W, labels, path=None, notebook=False):
    '''
    Creates a plot of the given graph. Layout is determined by t-SNE dimensionality reduction to 2d.
    Saves a plot of the 2d t-SNE of the embedding space.
    Can directly display plot if used within notebook.
    '''

    assert isinstance(edge_indices, torch.Tensor)
    assert isinstance(W, torch.Tensor)
    
    edge_indices = edge_indices.cpu()
    W = W.detach().cpu()

    num_nodes = edge_indices.max() + 1

    # compute list of edge index pairs from sparse adj matrix shape [2, num_edges] 
    edge_list = zip(*edge_indices.detach().tolist())
    edge_list = [(a,b) for a,b in edge_list if not a == b] # remove self loops

    # generate graph from edge list
    G = nx.from_edgelist(edge_list)

    ### PARAMETERS FOR DRAWING
    # node ids
    node_keys = range(num_nodes)
    def node_dict(x): return dict(zip(node_keys, x))

    # embedding mapping from Nd space to 2d
    W2D = TSNE(n_components=2).fit_transform(W)

    xs, ys = W2D.T.tolist()

    # node coordinates
    x_map = node_dict(xs)
    y_map = node_dict(ys)

    # node labels
    node_labels = node_dict(labels)

    # node sizes
    sizes = [12] * num_nodes
    size_map = node_dict(sizes)

    # node colours
    string_split = [string.split('_') for string in labels]
    if len(string_split[0]) == 1: # swat
        sensor_types = [str(*string)[:-3] for string in string_split]
    else: # wadi
        sensor_types = ['_'.join(string[:2]) for string in string_split]
    sensor_set = set(sensor_types)
    mapping = dict(zip(sensor_set, range(len(sensor_set))))
    node_color_map =  dict(zip(node_keys, [mapping[key] for key in sensor_types]))

    nx.set_node_attributes(G, node_labels, 'label')
    nx.set_node_attributes(G, node_color_map, 'group')
    nx.set_node_attributes(G, size_map, 'size')
    nx.set_node_attributes(G, x_map, 'x')
    nx.set_node_attributes(G, y_map, 'y')

    # pyvis network from networkx graph
    net = Network('1000px', '100%', bgcolor='#222222', font_color='white', notebook=notebook)
    # net = Network('1000px', '100%', bgcolor='#ffffff', font_color='black', notebook=notebook)
    net.from_nx(G)
    # gravity model for plot layout
    net.force_atlas_2based(gravity=-30, central_gravity=0.1, spring_length=50, spring_strength=0.001, damping=0.09, overlap=0.1)
    net.show_buttons(filter_=['physics'])
    if path is not None:
        net.save_graph(path)
    if notebook:
        net.show('graph.html')

    # plot of t-SNE
    _, axes = plt.subplots(1,2)
    for ax in axes:
        ax.scatter(xs, ys, c=list(node_color_map.values()), alpha=0.7)
    for i, label in enumerate(labels):
        axes[1].annotate(label, (xs[i], ys[i]))
    path = path.rsplit('.')[0] + '_tSNE.png'
    plt.savefig(path)


def plot_adjacency(A, labels, path=None, notebook=False):
    '''
    Creates a plot of the given graph. Layout is determined by t-SNE dimensionality reduction to 2d.
    Saves a plot of the 2d t-SNE of the embedding space.
    Can directly display plot if used within notebook.
    '''
    assert isinstance(A, torch.Tensor)
    
    A = A.detach()
    A.fill_diagonal_(0)
    A = A.cpu().numpy()

    num_nodes = A.shape[0]

    # generate graph from adjacency matrix
    directed = (A != A.T).any()
    if directed:
        G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    else:
        G = nx.from_numpy_matrix(A)

    ### PARAMETERS FOR DRAWING
    # node ids
    node_keys = range(num_nodes)
    def node_dict(x): return dict(zip(node_keys, x))

    # node labels
    node_labels = node_dict(labels)

    # node sizes
    sizes = [12] * num_nodes
    size_map = node_dict(sizes)

    # node colours
    string_split = [string.split('_') for string in labels]
    if len(string_split[0]) == 1: # swat
        sensor_types = [str(*string)[:-3] for string in string_split]
    else: # wadi
        sensor_types = ['_'.join(string[:2]) for string in string_split]
    sensor_set = set(sensor_types)
    mapping = dict(zip(sensor_set, range(len(sensor_set))))
    node_color_map =  dict(zip(node_keys, [mapping[key] for key in sensor_types]))

    nx.set_node_attributes(G, node_labels, 'label')
    nx.set_node_attributes(G, node_color_map, 'group')
    nx.set_node_attributes(G, size_map, 'size')

    # pyvis network from networkx graph
    directed = (A != A.T).any()
    net = Network('1000px', '100%', directed=directed, bgcolor='#222222', font_color='white', notebook=notebook)
    # net = Network('1000px', '100%', directed=directed, bgcolor='#ffffff', font_color='black', notebook=notebook)
    net.from_nx(G)
    # gravity model for plot layout
    net.force_atlas_2based(gravity=-30, central_gravity=0.1, spring_length=50, spring_strength=0.001, damping=0.09, overlap=0.1)
    net.show_buttons(filter_=['physics'])
    if path is not None:
        net.save_graph(path)
    if notebook:
        net.show('graph.html')