import networkx as nx
import numpy as np
import torch
from torch_geometric.nn import Node2Vec

from config import args
from utils import grid_params


def generate_graph(lat_grid_num, lng_grid_num):
    G = nx.grid_2d_graph(lat_grid_num, lng_grid_num, periodic=False)
    num_node = len(G.nodes)
    mapping = dict(zip(G.nodes(), range(num_node)))
    G = nx.relabel_nodes(G, mapping)
    return np.array(G.edges), num_node


def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(args.device), neg_rw.to(args.device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_epoch(model, loader, optimizer):
    # Training with epoch iteration
    last_loss = 1
    print("Training node embedding with node2vec...")
    for i in range(100):
        loss = train(model, loader, optimizer)
        print('Epoch: {0} \tLoss: {1:.4f}'.format(i, loss))
        if abs(last_loss - loss) < 1e-5:
            break
        else:
            last_loss = loss


@torch.no_grad()
def save_embeddings(model, num_nodes, dataset, device):
    model.eval()
    node_features = model(torch.arange(num_nodes, device=device)).cpu().numpy()
    np.save("./data/" + dataset + "/node_features.npy", node_features)
    print("Node embedding saved at: ./data/" + dataset + "/node_features.npy")


if __name__ == "__main__":

    if args.dataset == 'sz':
        boundary = {'min_lat': 22.48, 'max_lat': 22.58, 'min_lng': 113.9, 'max_lng': 114.1}
    elif args.dataset == 'cd':
        boundary = {'min_lat': 30.6, 'max_lat': 30.73, 'min_lng': 104, 'max_lng': 104.14}

    _, _, lat_grid_num, lng_grid_num = grid_params(boundary, args.grid_size)
    edge_index, num_node = generate_graph(lat_grid_num, lng_grid_num)

    edge_index = torch.LongTensor(edge_index).t().contiguous().to(args.device)

    model = Node2Vec(
        edge_index,
        embedding_dim=args.embedding_size,
        walk_length=20,
        context_size=10,
        walks_per_node=40,
        num_negative_samples=1,
        p=1,
        q=1,
        sparse=True,
        num_nodes=num_node
    ).to(args.device)

    loader = model.loader(batch_size=256, shuffle=True)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

    # Train until delta loss has been reached
    train_epoch(model, loader, optimizer)

    save_embeddings(model, num_node, args.dataset, args.device)
