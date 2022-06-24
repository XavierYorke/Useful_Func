#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2022/06/24 23:02:14
@Author  :   XavierYorke 
@Contact :   mzlxavier1230@gmail.com
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import time
import datetime
import math
from tqdm import tqdm, trange
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Modify file names in bulk')
    parser.add_argument('-d', '--dir', type=str, default='D:/影视/Eva', help='file dir')
    parser.add_argument('-o', '--output', type=str, default='', help='output dir')
    args = parser.parse_args()
    file_list = glob.glob(os.path.join(args.dir, '*.mkv'))
    # print(file_list[-1][:36] + '.mkv')
    for file_name in tqdm(file_list):
        new_name = file_name[:36] + '.mkv'
        # print(new_name)
        os.rename(file_name, new_name)