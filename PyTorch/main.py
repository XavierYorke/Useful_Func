#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# Time    :   2022/3/29
# Author  :   XavierYorke
# Contact :   mzlxavier1230@gmail.com

import argparse
import os
import time
import yaml
import torch
from .dataloaders import split_ds, train_transforms, val_transforms
from monai.data import Dataset, ThreadDataLoader
from .tools import setup_logger, train, set_seed
from .models import ResUnet, DiceLoss2
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default=None)
    parser.add_argument('-c', '--config', default='config.yaml')
    return parser.parse_args()


def main(epochs, batch_size, learning_rate, output_dir, log_step_freq, save_step_freq, seed, num_workers, buffer_size):
    set_seed(seed)

    # set output info
    output_dir = os.path.join(output_dir, time.strftime('%Y-%m-%d-%H-%M-%S'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save eval info
    with open(os.path.join(output_dir, 'eval.yaml'), 'w') as f:
        f.write('batch_size: {}\n'.format(batch_size))
        f.write('learning_rate: {}\n'.format(learning_rate))
        f.write('seed: {}\n'.format(seed))

    # set up log
    logger = setup_logger(output_dir)
    for data in str(config).split(', '):
        logger.info(data)

    # dataloader
    train_dict, val_dict = split_ds(data_config['dataset_dir'], 0.8)
    train_transforms.set_random_state(seed)
    val_transforms.set_random_state(seed)
    train_ds = Dataset(train_dict, train_transforms)
    val_ds = Dataset(val_dict, val_transforms)
    train_loader = ThreadDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                    buffer_size=buffer_size)
    val_loader = ThreadDataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                  buffer_size=buffer_size)

    # model
    model = ResUnet()
    start = 1
    if not args.resume is None:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        logger.info('successful load weights: {}'.format(args.resume))
        import re
        start = int(re.search('epoch-(\d+)', args.resume).group(1)) + 1
    model.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.loss_func = DiceLoss2()
    model.to(device)
    train(model, start, epochs, train_loader, val_loader, log_step_freq, save_step_freq, logger, device, output_dir)


if __name__ == '__main__':
    # set up multi-threading
    torch.multiprocessing.set_start_method('spawn')

    args = parse_args()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    data_config = config['data_config']
    train_config = config['train_config']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(**train_config)