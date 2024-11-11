import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import sys
from torch.utils.data import WeightedRandomSampler
import os
import numpy as np

sys.path.append('./')

from environments.cube3 import Cube3
from search_methods.astar import AStar
from utils import env_utils, nnet_utils, misc_utils
from argparse import ArgumentParser
from environments.environment_abstract import Environment, State

datas_validate = pd.read_csv('./search_methods/search_results/search_results7050.csv')

move_to_int = {
    'R': 10, 'R\'': 9, 'R2': 11,
    'L': 7, 'L\'': 6, 'L2': 8,
    'U': 1, 'U\'': 0, 'U2': 2,
    'D': 4, 'D\'': 3, 'D2': 5,
    'F': 16, 'F\'': 15, 'F2': 17,
    'B': 13, 'B\'': 12, 'B2': 14,
    'M': 19, 'M\'': 18, 'M2': 20,
    'S': 22, 'S\'': 21, 'S2': 23,
    'E': 25, 'E\'': 24, 'E2': 26
}

int_to_move = {v: k for k, v in move_to_int.items()}

