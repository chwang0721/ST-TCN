import os
import time
from functools import partial
from multiprocessing import Pool

import numpy as np

from config import args
from utils import get_time_period, build_time_index, select_time, build_grid_index, grid_params


# Calculate spatial distance between two points: m
def s_dist(traj_a, traj_b):
    traj_a, traj_b = np.radians(traj_a), np.radians(traj_b)
    dist_pairs = traj_a[:, None, :] - traj_b
    dlon = dist_pairs[:, :, 1]
    dlat = dist_pairs[:, :, 0]

    lat1 = traj_a[:, 0][:, None]
    lat2 = traj_b[:, 0][None, :]

    haver_formula = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

    r = 6371.0
    dist = 2 * r * np.arcsin(np.sqrt(haver_formula))
    return dist * 1000


# Calculate temporal distance between two points: s
def t_dist(traj_a, traj_b):
    return np.abs(traj_a[:, None] - traj_b)


# Calculate the st_diatance between two trajectories
def st_dist(traj_a, traj_b, threshold_d, threshold_t):
    start, end = select_time(traj_a, traj_b)

    if start <= end:
        loc_dists = s_dist(traj_a[:, :2], traj_b[:, :2])

        if np.min(loc_dists) <= threshold_d:
            time_dists = t_dist(traj_a[:, 3], traj_b[:, 3])
            dist = (np.exp(-loc_dists / threshold_d) + np.exp(-time_dists / threshold_t)) / 2

            score_a = np.max(dist, axis=1)
            score_b = np.max(dist, axis=0)

            indice_a = np.argsort(-score_a)[:args.contact_factor]
            indice_b = np.argsort(-score_b)[:args.contact_factor]

            score_a = score_a[indice_a]
            score_b = score_b[indice_b]

            if args.score_type == 's_t':
                score = (np.mean(score_a) + np.mean(score_b)) / 2
            elif args.score_type == 's_h':
                score = min(np.min(score_a), np.min(score_b))
            elif args.score_type == 's_l':
                theta = 2 * np.exp(-1)
                score = (len(score_a[score_a >= theta]) + len(score_b[score_b >= theta])) / (
                        len(score_a) + len(score_b))

            return score, indice_a.tolist(), indice_b.tolist()


# Parallel calcluate st_distance
def st_dist_parallel(idxs, trajs, trajs_num, time_index, grid_index):
    dist_pairs = []

    idxs, num = idxs[:-1], idxs[-1]

    for idx in idxs:
        traj_idx = np.array(trajs[idx])
        search_set = set(range(idx + 1, trajs_num))
        time_period = get_time_period(traj_idx)

        time_index_set = set()
        for time in time_period:
            time_index_set = time_index_set.union(time_index[time])

        grids = traj_idx[:, 4]
        grid_index_set = set()
        for grid in grids:
            grid_index_set = grid_index_set.union(grid_index[grid])

        search_set &= time_index_set
        search_set &= grid_index_set

        for j in search_set:
            result = st_dist(traj_idx, np.array(trajs[j]), args.threshold_d, args.threshold_t)
            if result:
                st_cor, indice_x, indice_y = result
                dist_pairs.append([idx, j, st_cor] + indice_x + indice_y)

    np.save('./temp_data/train_pairs_{}.npy'.format(num), np.array(dist_pairs))


def cal_top_k_gt(idxs, trajs, trajs_num, data_type, time_index, grid_index):
    pairs = []
    gt = []

    idxs, num = idxs[:-1], idxs[-1]

    for idx in idxs:
        dists = []
        idx_pairs = []
        traj_idx = np.array(trajs[idx])
        search_set = set(range(trajs_num)) - {idx}
        time_period = get_time_period(traj_idx)

        index_set = set()
        for time in time_period:
            index_set = index_set.union(time_index[time])

        grids = traj_idx[:, 4]
        grid_index_set = set()
        for grid in grids:
            grid_index_set = grid_index_set.union(grid_index[grid])

        search_set &= index_set
        search_set &= grid_index_set

        for j in search_set:
            result = st_dist(traj_idx, np.array(trajs[j]), args.threshold_d, args.threshold_t)
            if result:
                st_cor, _, _ = result
                dists.append([j, st_cor])
                idx_pairs.append([idx, j])

        if len(dists) >= 20:
            dists = np.array(dists)
            gt.append([idx, dists[np.argmax(dists[:, 1])][0]])
            pairs += idx_pairs

    np.save('./temp_data/{}_gt_{}.npy'.format(data_type, num), np.array(gt))
    np.save('./temp_data/{}_pairs_{}.npy'.format(data_type, num), np.array(pairs))


def merge(data_type):
    all_pairs = []
    path_list = os.listdir("./temp_data/")
    for file in path_list:
        if file[:3] == data_type[:3] and 'pairs' in file:
            pairs = np.load("./temp_data/" + file)
            all_pairs.append(pairs)

    if data_type == 'train':
        return np.concatenate(all_pairs, axis=0)

    else:
        all_gt = []
        for file in path_list:
            if file[:3] == data_type[:3] and 'gt' in file:
                gt = np.load("./temp_data/" + file)
                all_gt.append(gt)
        return np.concatenate(all_pairs, axis=0), np.concatenate(all_gt, axis=0)


def generate_params(trajs):
    index = build_time_index(trajs)
    trajs_num = len(trajs)
    idxs = np.array_split(range(trajs_num), args.cpu_num)
    return index, trajs_num, idxs


def main():
    print('------Generating Ground Truth------')
    print('Dataset:', args.dataset)

    train_trajs = np.load('./data/{}/train_trajs.npy'.format(args.dataset), allow_pickle=True)
    val_trajs = np.load('./data/{}/val_trajs.npy'.format(args.dataset), allow_pickle=True)
    test_trajs = np.load('./data/{}/test_trajs.npy'.format(args.dataset), allow_pickle=True)

    train_time_index, train_trajs_num, train_idxs = generate_params(train_trajs)
    val_time_index, val_trajs_num, val_idxs = generate_params(val_trajs)
    test_time_index, test_trajs_num, test_idxs = generate_params(test_trajs)

    train_grid_index = build_grid_index(train_trajs, lng_grid_num, grid_num)
    val_grid_index = build_grid_index(val_trajs, lng_grid_num, grid_num)
    test_grid_index = build_grid_index(test_trajs, lng_grid_num, grid_num)

    start = time.time()
    pool = Pool(processes=args.cpu_num)

    for idx in range(args.cpu_num):
        train_idxs[idx] = np.append(train_idxs[idx], idx)

    pool.map(partial(st_dist_parallel, trajs=train_trajs, trajs_num=train_trajs_num, time_index=train_time_index,
                     grid_index=train_grid_index), train_idxs)
    train_pairs = merge('train')
    print('Train pairs num:', len(train_pairs))
    np.save('./data/{}/{}/train_pairs_{}.npy'.format(args.dataset, args.score_type, args.contact_factor), train_pairs)

    for idx in range(args.cpu_num):
        val_idxs[idx] = np.append(val_idxs[idx], idx)
    pool.map(partial(cal_top_k_gt, trajs=val_trajs, trajs_num=val_trajs_num, data_type='val',
                     time_index=val_time_index, grid_index=val_grid_index), val_idxs)
    val_pairs, val_gt = merge('val')
    print('Validation pairs num:', len(val_pairs))
    np.save('./data/{}/{}/val_gt_{}.npy'.format(args.dataset, args.score_type, args.contact_factor), val_gt)
    np.save('./data/{}/{}/val_pairs_{}.npy'.format(args.dataset, args.score_type, args.contact_factor), val_pairs)

    for idx in range(args.cpu_num):
        test_idxs[idx] = np.append(test_idxs[idx], idx)
    pool.map(partial(cal_top_k_gt, trajs=test_trajs, trajs_num=test_trajs_num, data_type='test',
                     time_index=test_time_index, grid_index=test_grid_index), test_idxs)
    test_pairs, test_gt = merge('test')
    print('Test pairs num:', len(test_pairs))
    np.save('./data/{}/{}/test_gt_{}.npy'.format(args.dataset, args.score_type, args.contact_factor), test_gt)
    np.save('./data/{}/{}/test_pairs_{}.npy'.format(args.dataset, args.score_type, args.contact_factor), test_pairs)

    pool.close()
    pool.join()

    end = time.time()
    print('Time:', end - start)
    os.system('rm -r ./temp_data/*')


if __name__ == '__main__':
    if args.dataset == 'sz':
        boundary = {'min_lat': 22.48, 'max_lat': 22.58, 'min_lng': 113.9, 'max_lng': 114.1}
    elif args.dataset == 'cd':
        boundary = {'min_lat': 30.6, 'max_lat': 30.73, 'min_lng': 104, 'max_lng': 104.14}

    params = grid_params(boundary, args.grid_size)
    lat_size, lng_size, lat_grid_num, lng_grid_num = params
    grid_num = lat_grid_num * lng_grid_num
    main()
    print('Finished!')
    print('')
