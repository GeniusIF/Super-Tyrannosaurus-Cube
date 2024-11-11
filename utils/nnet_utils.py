from typing import List, Tuple, Optional
import numpy as np
import os
import torch
from torch import nn
from environments.environment_abstract import Environment, State
from collections import OrderedDict
import re
from random import shuffle
from torch import Tensor

import torch.optim as optim
from torch.optim.optimizer import Optimizer

from torch.multiprocessing import Queue, get_context

import time


# training
def states_nnet_to_pytorch_input(states_nnet: List[np.ndarray], device) -> List[Tensor]:
    states_nnet_tensors = []
    for tensor_np in states_nnet:
        tensor = torch.tensor(tensor_np, device=device)
        states_nnet_tensors.append(tensor)

    return states_nnet_tensors


def make_batches(states_nnet: List[np.ndarray],  outputs: np.ndarray,
                 batch_size: int) -> List[Tuple[List[np.ndarray], np.ndarray]]:
    num_examples = outputs.shape[0]
    rand_idxs = np.random.choice(num_examples, num_examples, replace=False)
    outputs = outputs.astype(np.float32)

    start_idx = 0
    batches = []
    while (start_idx + batch_size) <= num_examples:
        end_idx = start_idx + batch_size

        idxs = rand_idxs[start_idx:end_idx]

        inputs_batch = [x[idxs] for x in states_nnet]
        outputs_batch = outputs[idxs]

        batches.append((inputs_batch, outputs_batch))

        start_idx = end_idx

    return batches


# pytorch device
def get_device() -> Tuple[torch.device, List[int], bool]:
    device: torch.device = torch.device("cuda:0")
    devices: List[int] = get_available_gpu_nums()
    on_gpu: bool = False
    if devices and torch.cuda.is_available():
        device = torch.device("cuda:%i" % 1)
        on_gpu = False

    return device, devices, on_gpu


# loading nnet
def load_nnet(model_file: str, nnet: nn.Module, device: torch.device = None) -> nn.Module:
    # get state dict
    if device is None:
        state_dict = torch.load(model_file)
    else:
        state_dict = torch.load(model_file, map_location=device)

    # remove module prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = re.sub('^module\.', '', k)
        new_state_dict[k] = v

    # set state dict
    nnet.load_state_dict(new_state_dict, strict=False)

    nnet.eval()

    return nnet



def set_model_mode(model, train_branches=True):
    model.train()

    for name, layer in model.named_modules():
        if 'branch' in name:
            layer.train() 
        else:
            layer.eval() 


def get_heuristic_fn(nnet: nn.Module, device: torch.device, env: Environment, clip_zero: bool = False,
                     batch_size: Optional[int] = None, output_t: bool = False):
    set_model_mode(nnet)

    if not output_t:
        def heuristic_fn(states: List, is_nnet_format: bool = False) -> np.ndarray:
            cost_to_go = torch.zeros(0).to(device=device)
            if not is_nnet_format:
                num_states: int = len(states)
            else:
                num_states: int = states[0].shape[0]

            batch_size_inst: int = num_states
            if batch_size is not None:
                batch_size_inst = batch_size

            start_idx: int = 0

            while start_idx < num_states:
            # get batch
                end_idx: int = min(start_idx + batch_size_inst, num_states)

                # convert to nnet input
                if not is_nnet_format:
                    states_batch: List = states[start_idx:end_idx]
                    states_nnet_batch: List[np.ndarray] = env.state_to_nnet_input(states_batch)
                else:
                    states_nnet_batch = [x[start_idx:end_idx] for x in states]
                # get nnet output
                states_nnet_batch_tensors = states_nnet_to_pytorch_input(states_nnet_batch, device)
                output, gamma = nnet(*states_nnet_batch_tensors)


                cost_to_go_batch = output * gamma
                

                cost_to_go = torch.concatenate((cost_to_go, cost_to_go_batch[:, 0]), axis=0)

                start_idx: int = end_idx
            assert (cost_to_go.shape[0] == num_states)

            if clip_zero:
                cost_to_go = torch.maximum(cost_to_go, torch.tensor(0.0))

            return cost_to_go
    else:
        def heuristic_fn(states: List, is_nnet_format: bool = False) -> np.ndarray:
            time_to_go = torch.zeros(0).to(device=device)
            if not is_nnet_format:
                num_states: int = len(states)
            else:
                num_states: int = states[0].shape[0]

            batch_size_inst: int = num_states
            if batch_size is not None:
                batch_size_inst = batch_size

            start_idx: int = 0

            while start_idx < num_states:
            # get batch
                end_idx: int = min(start_idx + batch_size_inst, num_states)

                # convert to nnet input
                if not is_nnet_format:
                    states_batch: List = states[start_idx:end_idx]
                    states_nnet_batch: List[np.ndarray] = env.state_to_nnet_input(states_batch)
                else:
                    states_nnet_batch = [x[start_idx:end_idx] for x in states]
                # get nnet output
                states_nnet_batch_tensors = states_nnet_to_pytorch_input(states_nnet_batch, device)
                time = nnet(*states_nnet_batch_tensors)
                

                time_to_go = torch.concatenate((time_to_go, time[:, 0]), axis=0)

                start_idx: int = end_idx

            return time_to_go

    return heuristic_fn


def get_available_gpu_nums() -> List[int]:
    devices: Optional[str] = os.environ.get('CUDA_VISIBLE_DEVICES')
    return [int(x) for x in devices.split(',')] if devices else []


def load_heuristic_fn(nnet_dir: str, device: torch.device, on_gpu: bool, nnet: nn.Module, env: Environment,
                      clip_zero: bool = False, gpu_num: int = -1, batch_size: Optional[int] = None, output_t: bool = False):
    if (gpu_num >= 0) and on_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)

    model_file = "%s/model_state_dict.pt" % nnet_dir

    nnet = load_nnet(model_file, nnet, device=device)
    nnet.eval()
    nnet.to(device)
    if on_gpu:
       nnet = nn.DataParallel(nnet)

    heuristic_fn = get_heuristic_fn(nnet, device, env, clip_zero=clip_zero, batch_size=batch_size, output_t=output_t)

    return heuristic_fn
