#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# Time    :   2022/3/29
# Author  :   XavierYorke
# Contact :   mzlxavier1230@gmail.com

import logging
import os.path as osp


# log setting
def setup_logger(log_pth):
    log_level = logging.INFO
    format = '%(asctime)s %(levelname)s(%(lineno)d): %(message)s'
    datefmt = '%m-%d %H:%M:%S'
    # logfile = 'Netname-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = 'Netname.log'
    logfile = osp.join(log_pth, logfile)

    logging.basicConfig(level=log_level, format=format, datefmt=datefmt, filename=logfile)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(format))
    logging.getLogger().addHandler(console)
    return logging.getLogger()


if __name__ == '__main__':
    output_dir = r'output_dir'
    logger = setup_logger(output_dir)
    logger.info('START LOGGING')
