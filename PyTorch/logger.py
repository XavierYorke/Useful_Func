#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2022/03/04 00:37:28
@Author  :   XavierYorke 
@Contact :   mzlxavier1230@gmail.com
'''

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
    logger = logging.getLogger()
    return logger
