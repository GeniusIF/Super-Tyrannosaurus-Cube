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


datas_train = pd.read_csv('./data/cube3/train.csv')




parser: ArgumentParser = ArgumentParser()
parser.add_argument('--t_model_dir', type=str, required=True, help="Directory of t model")
parser.add_argument('--env', type=str, default='cube3', help="Environment: cube3")
parser.add_argument('--batch_size', type=int, default=1, help="Batch size for BWAS")
parser.add_argument('--weight', type=float, default=1.0, help="Weight of path cost")

parser.add_argument('--results_dir', type=str, required=True, help="Directory to save results")
parser.add_argument('--start_idx', type=int, default=0, help="")
parser.add_argument('--nnet_batch_size', type=int, default=None, help="Set to control how many states per GPU are "
                                                                        "evaluated by the neural network at a time. "
                                                                        "Does not affect final results, "
                                                                        "but will help if nnet is running out of "
                                                                        "memory.")

parser.add_argument('--verbose', action='store_true', default=False, help="Set for verbose")

args = parser.parse_args()


env: Environment = env_utils.get_environment(args.env)


model = env.get_nnet_model(output_t=True)

device, devices, on_gpu = nnet_utils.get_device()

heuristic_fn = nnet_utils.load_heuristic_fn(args.t_model_dir, device, on_gpu, model,
                                        env, clip_zero=True, batch_size=args.nnet_batch_size, output_t=True)


optimizer2 = optim.Adam(model.parameters_branch2(), lr=1e-4)



# Hyper-parameters
sample_ratio = 0.3
sample_gap = 10
num_epochs = 51
time_normalization_coef = 1e-3
margin = 0.1
time_alpha_initial = 1
time_alpha_ending = 10


loss_history = []

for epoch in range(num_epochs):
    time_total_loss = 0.0

    alpha = misc_utils.get_time_alpha(time_alpha_initial, time_alpha_ending, num_epochs, epoch)

    datas_train = datas_train.sample(frac=1)

    records_round = np.array([0.0, 0.0], dtype=float)

    for i in range(len(datas_train)):
        heuristic_tem_loss = 0.0
        time_tem_loss = 0.0

        data = datas_train.iloc[i]
        scramble_moves = data['scramble'].split(' ')
        known_solution_moves = data['recover'].split(' ')

        scramble_moves = [int(i) for i in scramble_moves]
        known_solution_moves = [int(i) for i in known_solution_moves]

        total_time = torch.tensor(data['time'], dtype=torch.float32) * time_normalization_coef

        initial_state = env.scramble(scramble_moves)


        astar = AStar([initial_state], env, heuristic_fn, [args.weight], known_solution_moves, device=device)

        
        t_expect = heuristic_fn(astar.known_solution_states[0:-1])
        N = t_expect.shape[0]
        time_loss = 0.5 * (t_expect[0] - total_time)**2

        weights = misc_utils.sample_weight_generator(N)
        n = 0

        while n < int(sample_ratio * N):
            sampler = WeightedRandomSampler(weights, 2)
            before, after = min(sampler), max(sampler)
            if after - before < sample_gap:
                continue
            time_loss += (alpha / N) * torch.max(torch.tensor(0.0), margin - t_expect[before] + t_expect[after])
            n += 1

        if time_loss.item() <= 1e+8:
            optimizer2.zero_grad()
            time_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters_branch2(), max_norm=1.0)
            optimizer2.step()

        time_tem_loss += time_loss.item()


        records_round += np.array([round(time_tem_loss, 4), 1], dtype=float)
        if (i+1) % 100 == 0 or i == len(datas_train) - 1:
            mean_time = records_round[0] / records_round[1]
            records_round = np.array([0.0, 0.0], dtype=float)

            loss_history.append({'epoch': epoch,
                                'step': i+1,
                                'time_loss': round(time_tem_loss, 4),
                                'mean time': round(mean_time.item(), 4)})
            
    
    if epoch % 5 == 0:
        print('#########################################################################')
        print('#########################################################################')
        print('#########################################################################')
        print('#########################################################################')

        model_path = os.path.join(args.results_dir, f't_model_epoch{epoch}.pt')
        torch.save(model.state_dict(), model_path)
        print(f'Saved model to {model_path}')

        loss_df = pd.DataFrame(loss_history)
        loss_csv_path = os.path.join(args.results_dir, 't_loss_records.csv')
        loss_df.to_csv(loss_csv_path, mode='a', header=not os.path.exists(loss_csv_path), index=False)
        print(f'Loss records saved to {loss_csv_path}')

        loss_history = []

