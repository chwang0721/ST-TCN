from collections import defaultdict

import networkx as nx
import numpy as np
import torch
from geopy.distance import geodesic
from numba import jit, prange
from torch_geometric.data import Data


# Get grid parameters
def grid_params(boundary, grid_size):
    lat_dist = geodesic((boundary['min_lat'], boundary['min_lng']), (boundary['max_lat'], boundary['min_lng'])).km
    lat_size = (boundary['max_lat'] - boundary['min_lat']) / lat_dist * grid_size

    lng_dist = geodesic((boundary['min_lat'], boundary['min_lng']), (boundary['min_lat'], boundary['max_lng'])).km
    lng_size = (boundary['max_lng'] - boundary['min_lng']) / lng_dist * grid_size

    lat_grid_num = int(lat_dist / grid_size) + 1
    lng_grid_num = int(lng_dist / grid_size) + 1
    return lat_size, lng_size, lat_grid_num, lng_grid_num


def build_time_index(trajs):
    time_index = defaultdict(set)

    for idx in range(len(trajs)):
        traj = np.array(trajs[idx])
        time_period = get_time_period(traj)

        for time in time_period:
            time_index[time].add(idx)
    return time_index


def build_grid_index(all_trajs, lng_grid_num, grid_num):
    grid_index = defaultdict(set)
    offsets = [1, -1, lng_grid_num, -lng_grid_num, lng_grid_num + 1, lng_grid_num - 1,
               -lng_grid_num + 1, -lng_grid_num - 1]

    for idx, traj in enumerate(all_trajs):
        grids = np.array(traj)[:, 4]

        for grid in grids:
            grid_index[grid].add(idx)
            for offset in offsets:
                i = grid + offset
                if 0 <= i < grid_num:
                    grid_index[i].add(idx)
    return grid_index


def get_time_period(traj):
    start_time = int(traj[0][3] / 1200)
    end_time = int(traj[-1][3] / 1200)

    time_period = set(np.arange(start_time, end_time + 1))
    return time_period


# Obtain contact start and end time
def select_time(traj_a, traj_b):
    start = max(traj_a[0][2], traj_b[0][2])
    end = min(traj_a[-1][2], traj_b[-1][2])

    return start, end


@jit(nopython=True, parallel=True)
def test_method(all_dist, test_trajs_len, ground_truth):
    hr_1, hr_5, hr_10 = 0, 0, 0
    cnt = 0

    for i in prange(test_trajs_len):
        mask = all_dist[:, 0] == i
        i_j_dist = all_dist[mask]
        if len(i_j_dist) > 20:
            dist = i_j_dist[:, -1]
            indices = np.argsort(-dist)

            i_j = i_j_dist[:, 1][indices]
            top_1_j, top_5_j, top_10_j = i_j[:1], i_j[:5], i_j[:10]

            gt_i = ground_truth[ground_truth[:, 0] == i]
            if len(gt_i) > 0:
                cnt += 1
                gt = gt_i.flatten()[1]

                hr_10 += len(np.intersect1d(top_10_j, gt))
                hr_5 += len(np.intersect1d(top_5_j, gt))
                hr_1 += len(np.intersect1d(top_1_j, gt))

    return float(hr_1) / cnt, float(hr_5) / cnt, float(hr_10) / cnt


def load_network(dataset, lat_grid_num, lng_grid_num):
    G = nx.grid_2d_graph(lat_grid_num, lng_grid_num, periodic=False)
    num_node = len(G.nodes)
    mapping = dict(zip(G.nodes(), range(num_node)))
    G = nx.relabel_nodes(G, mapping)
    edge_index = np.array(G.edges)

    node_embedding_path = "./data/" + dataset + "/node_features.npy"
    node_embeddings = np.load(node_embedding_path)

    edge_index = torch.LongTensor(edge_index).t().contiguous()
    node_embeddings = torch.tensor(node_embeddings, dtype=torch.float)

    road_network = Data(x=node_embeddings, edge_index=edge_index)

    return road_network
