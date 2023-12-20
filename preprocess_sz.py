import time
from datetime import datetime

import numpy as np
import pandas as pd
import transbigdata as tbd
from tqdm import tqdm
from utils import grid_params

from config import args


# Determine if the trajectory point are in boundary
def in_boundary(lat, lng, b):
    return b['min_lng'] < lng < b['max_lng'] and b['min_lat'] < lat < b['max_lat']


# Map trajectory point to gird
def grid_mapping(lat, lng, min_lat, min_lng, params):
    lat_size, lng_size, _, lng_grid_num = params
    grid_i = int((lat - min_lat) / lat_size)
    grid_j = int((lng - min_lng) / lng_size)
    return grid_i * lng_grid_num + grid_j


# Save preprocessed trajectories
def save_data(train_trajs, val_trajs, test_trajs, dataset):
    file_path = './data/{}/{}.npy'.format(dataset, '{}')
    np.save(file_path.format('train_trajs'), train_trajs)
    np.save(file_path.format('val_trajs'), val_trajs)
    np.save(file_path.format('test_trajs'), test_trajs)


def convert_time(timearray):
    timeArray = time.strptime(timearray, "%Y/%m/%d %H:%M:%S")
    timestamp = int(time.mktime(timeArray))
    timeslots = int(timestamp / 10)
    return timeslots, timestamp


def prepocess(data, dataset, boundary, params):
    shortest, longest = 20, 100
    data = data.sort_values(['Num', 'Time'])

    # obtain OD from trajectory data
    oddata = tbd.taxigps_to_od(data, col=['Num', 'Time', 'Lng', 'Lat', 'OpenStatus'])

    # extract deliver and idle trip trajectories
    data_deliver, _ = tbd.taxigps_traj_point(data, oddata, col=['Num', 'Time', 'Lng', 'Lat', 'OpenStatus'])

    data = data_deliver.groupby('ID')

    trajs = []
    total_points = 0
    for idx, value in tqdm(data, ncols=100):
        traj = np.array(value[['Lat', 'Lng', 'Time']])
        traj_seq = []
        valid = True

        if dataset == 'sz':
            last_timeslots, _ = convert_time('2013/10/22 ' + traj[0][2])
        elif dataset == 'cd':
            last_timeslots, _ = convert_time(traj[0][2])
        last_timeslots -= 1

        for lat, lng, timestamp in traj:
            if dataset == 'sz':
                timeslots, timestamp = convert_time('2013/10/22 ' + timestamp)
            elif dataset == 'cd':
                timeslots, timestamp = convert_time(timestamp)

            if in_boundary(lat, lng, boundary) and not timeslots == last_timeslots and timeslots - last_timeslots < 20:
                grid = grid_mapping(lat, lng, boundary['min_lat'], boundary['min_lng'], params)
                t = datetime.fromtimestamp(timestamp)
                timevec = [t.hour, t.minute, t.second, t.year, t.month, t.day]
                traj_seq.append([lat, lng, timeslots, timestamp, grid] + timevec)
            else:
                valid = False
                break

            last_timeslots = timeslots

        traj_len = len(traj_seq)
        if valid and shortest <= traj_len <= longest:
            trajs.append(traj_seq)
            total_points += traj_len

    print('Total trajectory num:', len(trajs))
    print('Total points num:', total_points)

    trajs = np.array(trajs, dtype=object)
    np.random.shuffle(trajs)
    train_trajs, val_trajs, test_trajs = np.split(trajs, [int(len(trajs) * 0.3), int(len(trajs) * 0.5)])
    return train_trajs, val_trajs, test_trajs


def main():
    # Read_data
    path = '../datasets/shenzhen/'
    data = pd.read_csv(path + 'TaxiData.txt', header=None, names=['Num', 'Time', 'Lng', 'Lat', 'OpenStatus', 'Speed'])

    train_trajs, val_trajs, test_trajs = prepocess(data, 'sz', boundary, params)
    save_data(train_trajs, val_trajs, test_trajs, 'sz')


if __name__ == '__main__':
    np.random.seed(1234)
    print('----------Preprocessing sz----------')
    boundary = {'min_lat': 22.48, 'max_lat': 22.58, 'min_lng': 113.9, 'max_lng': 114.1}
    params = grid_params(boundary, args.grid_size)
    main()
    print('Finished!')
    print('')
