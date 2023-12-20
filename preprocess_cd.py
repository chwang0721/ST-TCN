import numpy as np
import pandas as pd

from config import args
from preprocess_sz import prepocess, save_data
from utils import grid_params


def main():
    # Read_data
    path = '../datasets/chengdu/'
    data = pd.read_csv(path + '20140803_train.txt', header=None, names=['Num', 'Lat', 'Lng', 'OpenStatus', 'Time'])

    train_trajs, val_trajs, test_trajs = prepocess(data, 'cd', boundary, params)
    save_data(train_trajs, val_trajs, test_trajs, 'cd')


if __name__ == '__main__':
    np.random.seed(1234)
    print('----------Preprocessing cd----------')
    boundary = {'min_lat': 30.6, 'max_lat': 30.73, 'min_lng': 104, 'max_lng': 104.14}
    params = grid_params(boundary, args.grid_size)
    main()
    print('Finished!')
    print('')
