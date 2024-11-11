import sys
sys.path.append('./')

from typing import List, Tuple, Dict, Callable, Optional, Any
from environments.environment_abstract import Environment, State
import numpy as np
from heapq import heappush, heappop
import heapq
from subprocess import Popen, PIPE
import pandas as pd

from argparse import ArgumentParser
import torch
from utils import env_utils, nnet_utils, search_utils, misc_utils
import time
import os
import socket
from torch.multiprocessing import Process




class Node:
    __slots__ = ['state', 'path_cost', 'heuristic', 'cost', 'is_solved', 'parent_move', 'parent', 'transition_costs',
                 'children', 'bellman']

    def __init__(self, state: State, path_cost: float, is_solved: bool,
                 parent_move: Optional[int], parent):
        self.state: State = state
        self.path_cost: float = path_cost
        self.heuristic: Optional[float] = None
        self.cost: Optional[float] = None
        self.is_solved: bool = is_solved
        self.parent_move: Optional[int] = parent_move
        self.parent: Optional[Node] = parent

        self.transition_costs: List[float] = []
        self.children: List[Node] = []

        self.bellman: float = torch.inf

    def compute_bellman(self):
        if self.is_solved:
            self.bellman = 0.0
        elif len(self.children) == 0:
            self.bellman = self.heuristic
        else:
            for node_c, tc in zip(self.children, self.transition_costs):
                self.bellman = min(self.bellman, tc + node_c.heuristic)


OpenSetElem = Tuple[float, int, Node]


class Instance:

    def __init__(self, root_node: Node):
        self.open_set: List[OpenSetElem] = []
        self.heappush_count: int = 0
        self.closed_dict: Dict[State, float] = dict()
        self.popped_nodes: List[Node] = []
        self.goal_nodes: List[Node] = []
        self.num_nodes_generated: int = 0

        self.root_node: Node = root_node

        self.push_to_open([self.root_node])

    def push_to_open(self, nodes: List[Node]):
        for node in nodes:
            heappush(self.open_set, (node.cost, self.heappush_count, node))
            self.heappush_count += 1

    def pop_from_open(self, num_nodes: int, target_state: Optional[State] = None) -> List[Node]:
        if target_state is not None:

            index = None
            for i, (cost, count, node) in enumerate(self.open_set):
                if node.state == target_state:
                    index = i
                    break
            if index is not None:

                node = self.open_set[index][2]
                self.open_set[index] = self.open_set[-1]
                self.open_set.pop()
                if index < len(self.open_set):
                    heapq._siftup(self.open_set, index)
                    heapq._siftdown(self.open_set, 0, index)
                if node.is_solved:
                    self.goal_nodes.append(node)
                self.popped_nodes.append(node)
                return [node]
            else:
                raise ValueError("Target node not found in open set")
        else:
            num_to_pop: int = min(num_nodes, len(self.open_set))
            popped_nodes = [heappop(self.open_set)[2] for _ in range(num_to_pop)]
            self.goal_nodes.extend([node for node in popped_nodes if node.is_solved])
            self.popped_nodes.extend(popped_nodes)
            return popped_nodes

    def remove_in_closed(self, nodes: List[Node], target_state=None) -> List[Node]:
        nodes_not_in_closed: List[Node] = []

        for node in nodes:
            path_cost_prev: Optional[float] = self.closed_dict.get(node.state)
            if path_cost_prev is None:
                nodes_not_in_closed.append(node)
                self.closed_dict[node.state] = node.path_cost
            elif path_cost_prev > node.path_cost:
                nodes_not_in_closed.append(node)
                self.closed_dict[node.state] = node.path_cost
            elif target_state and node.state == target_state:
                nodes_not_in_closed.append(node)
                self.closed_dict[node.state] = node.path_cost

        return nodes_not_in_closed


def pop_from_open(instances: List[Instance], batch_size: int, target_state: Optional[State] = None) -> List[List[Node]]:
    popped_nodes_all: List[List[Node]] = []
    for instance in instances:
        popped_nodes = instance.pop_from_open(batch_size, target_state=target_state)
        popped_nodes_all.append(popped_nodes)
    return popped_nodes_all


def expand_nodes(instances: List[Instance], popped_nodes_all: List[List[Node]], env: Environment):
    # Get children of all nodes at once (for speed)
    popped_nodes_flat: List[Node]
    split_idxs: List[int]
    popped_nodes_flat, split_idxs = misc_utils.flatten(popped_nodes_all)

    if len(popped_nodes_flat) == 0:
        return [[]]

    states: List[State] = [x.state for x in popped_nodes_flat]

    states_c_by_node: List[List[State]]
    tcs_np: List[np.ndarray]

    states_c_by_node, tcs_np = env.expand(states)

    tcs_by_node: List[List[float]] = [list(x) for x in tcs_np]

    # Get is_solved on all states at once (for speed)
    states_c: List[State]

    states_c, split_idxs_c = misc_utils.flatten(states_c_by_node)
    is_solved_c: List[bool] = list(env.is_solved(states_c))
    is_solved_c_by_node: List[List[bool]] = misc_utils.unflatten(is_solved_c, split_idxs_c)

    # Update path costs for all states at once (for speed)
    parent_path_costs = np.expand_dims(np.array([node.path_cost for node in popped_nodes_flat]), 1)
    path_costs_c: List[float] = (parent_path_costs + np.array(tcs_by_node)).flatten().tolist()

    path_costs_c_by_node: List[List[float]] = misc_utils.unflatten(path_costs_c, split_idxs_c)

    # Reshape lists
    tcs_by_inst_node: List[List[List[float]]] = misc_utils.unflatten(tcs_by_node, split_idxs)
    patch_costs_c_by_inst_node: List[List[List[float]]] = misc_utils.unflatten(path_costs_c_by_node,
                                                                               split_idxs)
    states_c_by_inst_node: List[List[List[State]]] = misc_utils.unflatten(states_c_by_node, split_idxs)
    is_solved_c_by_inst_node: List[List[List[bool]]] = misc_utils.unflatten(is_solved_c_by_node, split_idxs)

    # Get child nodes
    instance: Instance
    nodes_c_by_inst: List[List[Node]] = []
    for inst_idx, instance in enumerate(instances):
        nodes_c_by_inst.append([])
        parent_nodes: List[Node] = popped_nodes_all[inst_idx]
        tcs_by_node: List[List[float]] = tcs_by_inst_node[inst_idx]
        path_costs_c_by_node: List[List[float]] = patch_costs_c_by_inst_node[inst_idx]
        states_c_by_node: List[List[State]] = states_c_by_inst_node[inst_idx]

        is_solved_c_by_node: List[List[bool]] = is_solved_c_by_inst_node[inst_idx]

        parent_node: Node
        tcs_node: List[float]
        states_c: List[State]
        str_reps_c: List[str]
        for parent_node, tcs_node, path_costs_c, states_c, is_solved_c in zip(parent_nodes, tcs_by_node,
                                                                              path_costs_c_by_node, states_c_by_node,
                                                                              is_solved_c_by_node):
            state: State
            for move_idx, state in enumerate(states_c):
                path_cost: float = path_costs_c[move_idx]
                is_solved: bool = is_solved_c[move_idx]
                node_c: Node = Node(state, path_cost, is_solved, move_idx, parent_node)

                nodes_c_by_inst[inst_idx].append(node_c)

                parent_node.children.append(node_c)

            parent_node.transition_costs.extend(tcs_node)

        instance.num_nodes_generated += len(nodes_c_by_inst[inst_idx])

    return nodes_c_by_inst


def remove_in_closed(instances: List[Instance], nodes_c_all: List[List[Node]], target_state=None) -> List[List[Node]]:
    for inst_idx, instance in enumerate(instances):
        nodes_c_all[inst_idx] = instance.remove_in_closed(nodes_c_all[inst_idx], target_state=target_state)

    return nodes_c_all


def add_heuristic_and_cost(nodes: List[Node], heuristic_fn: Callable,
                           weights: List[float], device="cuda:0") -> Tuple[np.ndarray, np.ndarray]:

    # flatten nodes
    nodes: List[Node]

    if len(nodes) == 0:
        return torch.zeros(0), torch.zeros(0, device=device)

    # get heuristic
    states: List[State] = [node.state for node in nodes]

    # compute node cost
    heuristics = heuristic_fn(states)

    path_costs = torch.tensor([node.path_cost for node in nodes], device=device)
    is_solved = torch.tensor([node.is_solved for node in nodes], device=device)

    costs = torch.tensor(weights, device=device) * path_costs + heuristics * torch.logical_not(is_solved).to(device)

    # add cost to node
    for node, heuristic, cost in zip(nodes, heuristics, costs):
        node.heuristic = heuristic
        node.cost = cost

    return path_costs, heuristics


def add_to_open(instances: List[Instance], nodes: List[List[Node]]) -> None:
    nodes_inst: List[Node]
    instance: Instance
    for instance, nodes_inst in zip(instances, nodes):
        instance.push_to_open(nodes_inst)


def get_path(node: Node) -> Tuple[List[State], List[int], float]:
    path: List[State] = []
    moves: List[int] = []

    parent_node: Node = node
    while parent_node.parent is not None:
        path.append(parent_node.state)

        moves.append(parent_node.parent_move)
        parent_node = parent_node.parent

    path.append(parent_node.state)

    path = path[::-1]
    moves = moves[::-1]

    return path, moves, node.path_cost


class AStar:

    def __init__(self, states: List[State], env: Environment, heuristic_fn: Callable, weights: List[float], known_solution_moves: List[int] = None, device = 'cuda:0'):
        self.env: Environment = env
        self.weights: List[float] = weights
        self.step_num: int = 0
        self.device = device

        self.timings: Dict[str, float] = {"pop": 0.0, "expand": 0.0, "check": 0.0, "heur": 0.0,
                                          "add": 0.0, "itr": 0.0}

        # compute starting costs
        root_nodes: List[Node] = []
        is_solved_states = self.env.is_solved(states)
        for state, is_solved in zip(states, is_solved_states):
            root_node: Node = Node(state, 0.0, is_solved, None, None)
            root_nodes.append(root_node)

        add_heuristic_and_cost(root_nodes, heuristic_fn, self.weights, device=device)

        # initialize instances
        self.instances: List[Instance] = []
        for root_node in root_nodes:
            self.instances.append(Instance(root_node))


        self.known_solution_states = None
        if known_solution_moves:

            self.known_solution_states: List[State] = [states[0]]  # 起始状态
            current_state = states[0]
            for move in known_solution_moves:
                next_states, _ = env.next_state([current_state], move)
                current_state = next_states[0]
                self.known_solution_states.append(current_state)
            self.known_solution_states.append(None)
            self.solution_index = 0

    def step(self, heuristic_fn: Callable, batch_size: int, include_solved: bool = False, verbose: bool = False):
        start_time_itr = time.time()
        instances: List[Instance]
        if include_solved:
            instances = self.instances
        else:
            instances = [instance for instance in self.instances if len(instance.goal_nodes) == 0]

        # Pop from open
        start_time = time.time()

        target_state = self.known_solution_states[self.solution_index] if self.known_solution_states else None
        if self.known_solution_states:
            self.solution_index += 1
        popped_nodes_all: List[List[Node]] = pop_from_open(instances, batch_size, target_state=target_state)
        pop_time = time.time() - start_time

        # Expand nodes
        start_time = time.time()
        nodes_c_all: List[List[Node]] = expand_nodes(instances, popped_nodes_all, self.env)
        expand_time = time.time() - start_time

        # Get heuristic of children, do heur before check so we can do backup
        start_time = time.time()
        nodes_c_all_flat, _ = misc_utils.flatten(nodes_c_all)
        weights, _ = misc_utils.flatten([[weight] * len(nodes_c) for weight, nodes_c in zip(self.weights, nodes_c_all)])
        path_costs, heuristics = add_heuristic_and_cost(nodes_c_all_flat, heuristic_fn, weights)
        heur_time = time.time() - start_time

        # Check if children are in closed
        start_time = time.time()
        target_state = self.known_solution_states[self.solution_index] if self.known_solution_states else None
        nodes_c_all = remove_in_closed(instances, nodes_c_all, target_state=target_state)
        check_time = time.time() - start_time

        # Add to open
        start_time = time.time()
        add_to_open(instances, nodes_c_all)
        add_time = time.time() - start_time

        itr_time = time.time() - start_time_itr



        # Print to screen
        if verbose:
            if heuristics.shape[0] > 0:
                min_heur = np.min(heuristics)
                min_heur_pc = path_costs[np.argmin(heuristics)]
                max_heur = np.max(heuristics)
                max_heur_pc = path_costs[np.argmax(heuristics)]

                print("Itr: %i, Added to OPEN - Min/Max Heur(PathCost): "
                      "%.2f(%.2f)/%.2f(%.2f) " % (self.step_num, min_heur, min_heur_pc, max_heur, max_heur_pc))

            print("Times - pop: %.2f, expand: %.2f, check: %.2f, heur: %.2f, "
                  "add: %.2f, itr: %.2f" % (pop_time, expand_time, check_time, heur_time, add_time, itr_time))

            print("")

        # Update timings
        self.timings['pop'] += pop_time
        self.timings['expand'] += expand_time
        self.timings['check'] += check_time
        self.timings['heur'] += heur_time
        self.timings['add'] += add_time
        self.timings['itr'] += itr_time

        self.step_num += 1

        return

    def has_found_goal(self) -> List[bool]:
        goal_found: List[bool] = [len(self.get_goal_nodes(idx)) > 0 for idx in range(len(self.instances))]

        return goal_found

    def get_goal_nodes(self, inst_idx) -> List[Node]:
        return self.instances[inst_idx].goal_nodes

    def get_goal_node_smallest_path_cost(self, inst_idx) -> Node:
        goal_nodes: List[Node] = self.get_goal_nodes(inst_idx)
        path_costs: List[float] = [node.path_cost for node in goal_nodes]

        goal_node: Node = goal_nodes[int(np.argmin(path_costs))]

        return goal_node

    def get_num_nodes_generated(self, inst_idx: int) -> int:
        return self.instances[inst_idx].num_nodes_generated

    def get_popped_nodes(self) -> List[List[Node]]:
        popped_nodes_all: List[List[Node]] = [instance.popped_nodes for instance in self.instances]
        return popped_nodes_all


def main():
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--gamma_model_dir', type=str, required=True, help="Directory of gamma model")
    parser.add_argument('--t_model_dir', type=str, required=True, help="Directory of t model")
    parser.add_argument('--results_dir', type=str, required=True, help="Directory of search results")


    parser.add_argument('--env', type=str, default='cube3', help="Environment: cube3")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for BWAS")
    parser.add_argument('--weight', type=float, default=1.0, help="Weight of path cost")
    parser.add_argument('--language', type=str, default="python", help="python or cpp")

    parser.add_argument('--start_idx', type=int, default=0, help="")
    parser.add_argument('--nnet_batch_size', type=int, default=None, help="Set to control how many states per GPU are "
                                                                          "evaluated by the neural network at a time. "
                                                                          "Does not affect final results, "
                                                                          "but will help if nnet is running out of "
                                                                          "memory.")

    parser.add_argument('--verbose', action='store_true', default=False, help="Set for verbose")
    parser.add_argument('--debug', action='store_true', default=False, help="Set when debugging")

    args = parser.parse_args()



    # environment
    env: Environment = env_utils.get_environment(args.env)


    # get data   
    datas_test = pd.read_csv('./data/cube3/test.csv')['scramble']
    datas_train = pd.read_csv('./data/cube3/train.csv')
    states_list = []
    for i in range(len(datas_test)):
        data = datas_test.iloc[i]
        moves = [int(_) for _ in data.split(' ')]
        state = env.scramble(moves)
        states_list.append(state)
    states: List[State] = states_list[args.start_idx:]


    # initialize results
    results: Dict[str, Any] = dict()
    results["states"] = states

    with torch.no_grad():
        if args.language == "python":
            solns, times, steps = bwas_python(args, env, states)

    # get loop net
    device, _, _ = nnet_utils.get_device()
    
    loop_dict = [1, 0,  2,
            4,  3,  5,
            7,  6,  8, 
            10, 9,  11, 
            13, 12, 14, 
            16, 15, 17, 
            19, 18, 20, 
            22, 21, 23, 
            25, 24, 26]

    loop_policy = torch.zeros(28).to(device)
    sample_num = 0
    for i in range(len(datas_train)):
        data = datas_train.iloc[i]
        scramble_moves = data['scramble'].split(' ')
        known_solution_moves = data['recover'].split(' ')

        scramble_moves = [int(i) for i in scramble_moves]
        known_solution_moves = [int(i) for i in known_solution_moves]

        tem_state = [env.scramble(scramble_moves)]

        for j in range(len(known_solution_moves) - 1):
            move1, move2 = known_solution_moves[j], known_solution_moves[j+1]
            if loop_dict[move1] == move2:
                loop_policy[move1] += 1.0
            else:
                loop_policy[-1] += 1.0
            sample_num += 1
            
            tem_state, _ = env.next_state(tem_state, move1)
    loop_policy = loop_policy / sample_num
    
    solns_new = []
    for i in range(len(solns)):
        soln_new = []
        tem_state = [states[i]]
        for move in solns[i]:
            while True:
                picked_actions = int(torch.distributions.Categorical(loop_policy).sample())
                if picked_actions != 27:
                    soln_new.append(picked_actions)
                    soln_new.append(loop_dict[picked_actions])
                    steps[i] += 2
                else:
                    break
            soln_new.append(move)
            tem_state, _ = env.next_state(tem_state, move)
        solns_new.append(soln_new)



    results["solutions"] = solns_new
    results["times"] = times.tolist()
    results["steps"] = steps
    print(results)

    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(args.results_dir, 'search_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f'Search records saved to {results_csv_path}')


def bwas_python(args, env: Environment, states: List[State]):
    # get device
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()

    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    gamma_heuristic_fn = nnet_utils.load_heuristic_fn(args.gamma_model_dir, device, on_gpu, env.get_nnet_model(),
                                                env, clip_zero=True, batch_size=args.nnet_batch_size)
    
    t_heuristic_fn = nnet_utils.load_heuristic_fn(args.t_model_dir, device, on_gpu, env.get_nnet_model(output_t=True),
                                                env, clip_zero=True, batch_size=args.nnet_batch_size, output_t=True)

    solns: List[List[int]] = []
    times: List[float] = []
    steps: List[int] = []

    times = t_heuristic_fn(states)

    for state_idx, state in enumerate(states):

        num_itrs: int = 0
        astar = AStar([state], env, gamma_heuristic_fn, [args.weight])
        while not min(astar.has_found_goal()):
            astar.step(gamma_heuristic_fn, args.batch_size, verbose=args.verbose)
            num_itrs += 1


        soln: List[int]

        goal_node: Node = astar.get_goal_node_smallest_path_cost(0)
        _, soln, _ = get_path(goal_node)

        # record solution information
        solns.append(soln)

        # check soln
        assert search_utils.is_valid_soln(state, soln, env)

        steps.append(len(soln))

    return solns, times, steps

if __name__ == "__main__":
    main()
