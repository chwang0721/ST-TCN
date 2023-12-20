import numpy as np
import torch
from torch.utils.data import DataLoader

from config import args
from data_loader import MyDataset, collater_train, collater_test
from logging_set import get_logger
from trainer import Trainer
from utils import grid_params, test_method


def hr_test(model, trajs_len, ground_truth, logger):
    all_dist = model.test()
    hr_1, hr_5, hr_10 = test_method(all_dist, trajs_len, ground_truth)
    logger.info(f'HR@1: {hr_1:.5f}\tHR@5: {hr_5:.5f}\tHR@10: {hr_10:.5f}')
    return hr_1


def main():
    if args.mode == 'train':
        train_trajs = np.load(f'./data/{args.dataset}/train_trajs.npy', allow_pickle=True)
        val_trajs = np.load(f'./data/{args.dataset}/val_trajs.npy', allow_pickle=True)
        val_trajs_len = len(val_trajs)

        train_data = np.load(f'./data/{args.dataset}/{args.score_type}/train_pairs_{args.contact_factor}.npy')
        val_data = np.load(f'./data/{args.dataset}/{args.score_type}/val_pairs_{args.contact_factor}.npy')
        ground_truth = np.load(f'./data/{args.dataset}/{args.score_type}/val_gt_{args.contact_factor}.npy')

        collate_fn_train = collater_train(train_trajs, args.contact_factor)
        collate_fn_val = collater_test(val_trajs)

        train_data = MyDataset(train_data)
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_fn_train, num_workers=20)

        val_data = MyDataset(val_data)
        val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_fn_val, num_workers=20)

        Trainer_contact = Trainer(train_loader, val_loader, args.embedding_size, args.device,
                                  args.dataset, args.lr, lat_grid_num, lng_grid_num, args.contact_factor, args.score_type)

        hr = 0
        logger = get_logger(Trainer_contact.train_log_path)
        logger.info('Start training!')
        logger.info(f'Number of points used to calculate the score: {args.contact_factor}')

        for epoch in range(args.epochs):
            loss = Trainer_contact.train()
            logger.info(f'Epoch: [{epoch + 1}/{args.epochs}]\tLoss: {loss:.5f}')
            hr_1 = hr_test(Trainer_contact, val_trajs_len, ground_truth, logger)
            if hr_1 > hr:
                torch.save(Trainer_contact.model.state_dict(), Trainer_contact.model_path)
                hr = hr_1

        logger.info('Finish training!')

    elif args.mode == 'test':

        test_trajs = np.load(f'./data/{args.dataset}/test_trajs.npy', allow_pickle=True)
        test_data = np.load(f'./data/{args.dataset}/{args.score_type}/test_pairs_{args.contact_factor}.npy')
        ground_truth = np.load(f'./data/{args.dataset}/{args.score_type}/test_gt_{args.contact_factor}.npy')
        test_trajs_len = len(test_trajs)

        test_data = MyDataset(test_data)

        collate_fn_test = collater_test(test_trajs)
        test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_fn_test, num_workers=20)

        train_loader = None

        Trainer_contact = Trainer(train_loader, test_loader, args.embedding_size, args.device, args.dataset, args.lr,
                                  lat_grid_num, lng_grid_num, args.contact_factor, args.score_type)

        logger = get_logger(Trainer_contact.test_log_path)
        logger.info('Start testing!')
        logger.info(f'Number of points used to calculate the score: {args.contact_factor}')

        Trainer_contact.model.load_state_dict(torch.load(Trainer_contact.model_path))
        hr_test(Trainer_contact, test_trajs_len, ground_truth, logger)

        logger.info('Finish testing!')


if __name__ == "__main__":

    if args.dataset == 'sz':
        boundary = {'min_lat': 22.48, 'max_lat': 22.58, 'min_lng': 113.9, 'max_lng': 114.1}
    elif args.dataset == 'cd':
        boundary = {'min_lat': 30.6, 'max_lat': 30.73, 'min_lng': 104, 'max_lng': 104.14}

    print('=========================================')
    print('Dataset:', args.dataset)

    params = grid_params(boundary, args.grid_size)
    _, _, lat_grid_num, lng_grid_num = params

    main()
    print('')
