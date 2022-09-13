#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Time    :   2022/03/04 00:37:28
@Author  :   XavierYorke 
@Contact :   mzlxavier1230@gmail.com
"""

import logging
import os.path as osp


# log setting
def setup_logger(log_pth):
    Log_level = logging.INFO
    Format = '%(asctime)s %(levelname)s(%(lineno)d): %(message)s'
    Datefmt = '%m-%d %H:%M:%S'
    # logfile = 'Netname-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    Logfile = 'Netname.log'
    Logfile = osp.join(log_pth, Logfile)

    logging.basicConfig(level=Log_level, format=Format, datefmt=Datefmt, filename=Logfile)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(Format))
    logging.getLogger().addHandler(console)
    logger = logging.getLogger()
    return logger
