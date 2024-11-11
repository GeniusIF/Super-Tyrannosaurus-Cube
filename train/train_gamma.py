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
parser.add_argument('--gamma_model_dir', type=str, required=True, help="Directory of gamma model")
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


model = env.get_nnet_model()

device, devices, on_gpu = nnet_utils.get_device()

heuristic_fn = nnet_utils.load_heuristic_fn(args.gamma_model_dir, device, on_gpu, model,
                                        env, clip_zero=True, batch_size=args.nnet_batch_size)


optimizer1 = optim.Adam(model.parameters_branch1(), lr=1e-4)



# Hyper-parameters
init_epoch = 0 # 0 for initial training
num_epochs = 81

loss_history = []

for epoch in range(init_epoch, init_epoch + num_epochs):
    datas_train = datas_train.sample(frac=1)

    records_round = np.array([0.0, 0.0], dtype=float)

    for i in range(len(datas_train)):
        heuristic_tem_loss = 0.0

        data = datas_train.iloc[i]
        scramble_moves = data['scramble'].split(' ')
        known_solution_moves = data['recover'].split(' ')

        scramble_moves = [int(i) for i in scramble_moves]
        known_solution_moves = [int(i) for i in known_solution_moves]


        initial_state = env.scramble(scramble_moves)


        astar = AStar([initial_state], env, heuristic_fn, [args.weight], known_solution_moves, device=device)
        astar.step(heuristic_fn, args.batch_size, verbose=args.verbose)

        loss_count = 0
        
        while not min(astar.has_found_goal()) and astar.solution_index < len(astar.known_solution_states) - 1:
            successor_set = astar.instances[0].open_set
            successor_states = [successor[-1].state for successor in successor_set]
            successor_costs = heuristic_fn(successor_states)

            target_state = astar.known_solution_states[astar.solution_index]
            target_index = successor_states.index(target_state)

            neg_f_values = - successor_costs
            probabilities = F.softmax(neg_f_values, dim=0)


            heuristic_loss = -torch.log(probabilities[target_index])  
            if heuristic_loss.item() <= 1e+10:
                optimizer1.zero_grad()
                heuristic_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters_branch1(), max_norm=1.0)
                optimizer1.step()
                heuristic_tem_loss += heuristic_loss.item()
                loss_count += 1

            astar.step(heuristic_fn, args.batch_size, verbose=args.verbose)

        if loss_count == 0:
            print(heuristic_loss.item())
            print("Error!! loss_count==0")
            continue


        

        records_round += np.array([round(heuristic_tem_loss / loss_count, 4), 1], dtype=float)
        if (i+1) % 100 == 0 or i == len(datas_train) - 1:
            mean_heuristic = records_round[0] / records_round[1]
            records_round = np.array([0.0, 0.0], dtype=float)

            print('######################################################################')
            print(f'data {i+1}, heuristic loss:     {round(heuristic_tem_loss / loss_count, 4)}')
            print(f'data {i+1}, mean heuristic:     {round(mean_heuristic.item(), 4)}')

            loss_history.append({'epoch': epoch,
                                'step': i+1,
                                'heuristic_loss': round(heuristic_tem_loss / loss_count, 4),
                                'mean heuristic': round(mean_heuristic.item(), 4)})


    print('#########################################################################')
    print('#########################################################################')
    print('#########################################################################')
    print('#########################################################################')
    
    model_path = os.path.join(args.results_dir, f'gamma_model_epoch{epoch}.pt')
    torch.save(model.state_dict(), model_path)
    print(f'Saved model to {model_path}')

    loss_df = pd.DataFrame(loss_history)
    loss_csv_path = os.path.join(args.results_dir, 'gamma_loss_records.csv')
    loss_df.to_csv(loss_csv_path, mode='a', header=not os.path.exists(loss_csv_path), index=False)
    print(f'Loss records saved to {loss_csv_path}')

    loss_history = []
