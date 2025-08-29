import csv
import os
import time
from datetime import datetime
import random
import yaml

import matplotlib.pyplot as plt
import minari
import numpy as np
import torch

import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

from drone_env import DroneEnv
from car_env import CarEnv

from tqdm import tqdm

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from train_diffusion_policy import init_noise_pred_net
# from train_dipper import init_dipper_net  # NOT IN USE

from policies.fm_policy import DiffusionSampler
from policies.uniform_policy import UniformSampler

from planners.RRT import RRT_Planner
from planners.random_tree import RandomTreePlanner
from planners.MPC import MPC_Planner

import common.map_utils


# import gdown

# Function to run the experiment
def run_experiment(planner, num_runs=100):
    success_count = 0
    runtimes = []
    iterations = []
    path_lengths = []
    path_avg_speeds = []
    number_of_nodes = []

    for _ in range(num_runs):
        planner.reset()
        path, actions = planner.plan()
        curr_iterations = planner.results["iterations"]
        runtime = planner.results["time"]
        num_nodes = planner.results["number_of_nodes"]

        if path is not None:
            path_length = 0
            for i in range(len(path) - 1):
                path_length += np.linalg.norm(path[i + 1, :2] - path[i, :2])
            avg_speed = np.mean(np.linalg.norm(path[:, 3:5], axis=1))

            success_count += 1
        runtimes.append(runtime)
        iterations.append(curr_iterations)
        path_lengths.append(path_length)
        path_avg_speeds.append(avg_speed)
        number_of_nodes.append(num_nodes)

    success_rate = success_count / num_runs
    return success_rate, runtimes, iterations, path_lengths, path_avg_speeds, number_of_nodes


def evaluate_all_scenarios(mazes_dir, scenarios_file, cfg_file, total_runs=100, time_budget=60, diffusion_sampler_checkpoints=None):
    # Settings

    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    unet_dims = {
        'small': [64, 128, 256],
        'medium': [256, 512, 1024],
        'large': [512, 1024, 2048],
        'xlarge': [1024, 2048, 4096]
    }

    with open(f"cfgs/{cfg_file}.yaml", "r") as file:
        loaded_config = yaml.safe_load(file)

    debug = loaded_config.get('debug', False)
    prediction_type = loaded_config.get('prediction_type', "actions")  # ["actions","observations"]
    obs_history = loaded_config.get('obs_history', 1)
    action_history = loaded_config.get('action_history', 1)
    position_conditioned = False
    goal_conditioned = loaded_config.get('goal_conditioned', True)
    local_map_conditioned = loaded_config.get('local_map_conditioned', True)
    local_map_size = loaded_config.get('local_map_size', 20)
    local_map_scale = loaded_config.get('local_map_scale', 0.2)
    local_map_embedding_dim = loaded_config.get('local_map_embedding_dim', 400)
    env_id = loaded_config.get('env_id', "carmaze")
    policy = loaded_config.get('policy', "flow_matching")  # ["diffusion","flow_matching"]
    num_diffusion_iters = loaded_config.get('planning_diffusion_iters', 5)
    unet_down_dims = unet_dims[loaded_config.get('denoiser_size', 'large')]
    pred_horizon = loaded_config.get('pred_horizon', 64)
    action_horizon = loaded_config.get('action_horizon', 8)
    goal_conditioning_bias = loaded_config.get('goal_conditioning_bias', 0.85)
    prop_duration = loaded_config.get('prop_duration', [64])


    goal_dim = 2
    s_global=1.0
    if "antmaze" in env_id.lower():
        obs_dim = 31    # 6D rotation representation
        if not position_conditioned:
            obs_dim -= 2  # remove (x,y)
        action_dim = 8
        s_global = 4.0
        action_horizon = 2
        local_map_scale = 0.8
    elif "pointmaze" in env_id.lower():
        obs_dim = 4
        if not position_conditioned:
            obs_dim -= 2  # remove (x,y)
        action_dim = 2
    elif "dronemaze" in env_id.lower():
        obs_dim = 10
        if not position_conditioned:
            obs_dim -= 2  # remove (x,y)
        action_dim = 4
    elif "car" in env_id.lower():
        full_obs_dim = 6
        obs_dim = full_obs_dim
        if not position_conditioned:
            obs_dim -= 3  # remove (x,y,theta)
        action_dim = 2
    else:
        raise ValueError(f"Invalid env_id: {env_id}")

    render_mode = 'human' if debug else 'rgb_array'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = 'checkpoints/'

    noise_scheduler = DDPMScheduler(num_train_timesteps=num_diffusion_iters, beta_schedule='squaredcos_cap_v2',
                                    clip_sample=True, prediction_type='epsilon')

    noise_pred_net = init_noise_pred_net(
        input_dim=action_dim if prediction_type == "actions" else full_obs_dim,
        action_dim=action_dim,
        obs_dim=obs_dim,
        obs_history=obs_history,
        action_history=action_history,
        goal_conditioned=goal_conditioned,
        goal_dim=goal_dim,
        local_map_conditioned=local_map_conditioned,
        local_map_encoder="resnet",
        local_map_embedding_dim=local_map_embedding_dim,
        local_map_size=local_map_size,
        down_dims=unet_down_dims,
    )

    checkpoint = torch.load(output_dir + diffusion_sampler_checkpoints['resnet'])
    noise_pred_net.load_state_dict(checkpoint['noise_pred_net_state_dict'])
    noise_pred_net = noise_pred_net.to(device).eval()

    diffusion_sampler_small_resnet = DiffusionSampler(noise_pred_net, noise_scheduler, env_id,
                                                      policy='flow_matching',
                                                      pred_horizon=pred_horizon, action_dim=action_dim,
                                                      prediction_type=prediction_type,
                                                      obs_history=obs_history, action_history=action_history,
                                                      goal_conditioned=True, num_diffusion_iters=num_diffusion_iters,
                                                      local_map_size=local_map_size).eval()


    os.makedirs('benchmark_results', exist_ok=True)

    # Load the scenarios from the CSV file
    with open(scenarios_file, mode='r') as scenarios_csv:
        scenarios_reader = csv.reader(scenarios_csv)
        next(scenarios_reader)  # Skip the header

        # Iterate through all scenarios
        for scenario in scenarios_reader:
            if "car" in env_id.lower():
                scenario_name, maze_name, start_row, start_col, start_deg, goal_row, goal_col = scenario
            else:
                scenario_name, maze_name, start_row, start_col, goal_row, goal_col = scenario

            # Load the corresponding maze
            maze_path = os.path.join(mazes_dir, f'{maze_name}.csv')
            if not os.path.exists(maze_path):
                print(f"Maze file {maze_path} not found, skipping scenario.")
                continue

            maze_data = np.loadtxt(maze_path, delimiter=',')

            # Create a new environment for each maze with the maze data
            if "pointmaze" in env_id.lower():
                env = gym.make('PointMaze_Large-v3', maze_map=maze_data, render_mode=render_mode)
                start_xy = env.maze.cell_rowcol_to_xy(np.array([int(start_row), int(start_col)]))
                goal_xy = env.maze.cell_rowcol_to_xy(np.array([int(goal_row), int(goal_col)]))
                start = np.array([start_xy[0], start_xy[1], 0.0, 0.0])
                goal = np.array([goal_xy[0], goal_xy[1], 0.0, 0.0])
            elif "antmaze" in env_id.lower():
                env = gym.make('AntMaze_Large-v4', maze_map=maze_data, render_mode=render_mode)
                start_xy = env.maze.cell_rowcol_to_xy(np.array([int(start_row), int(start_col)]))
                goal_xy = env.maze.cell_rowcol_to_xy(np.array([int(goal_row), int(goal_col)]))
                start = np.zeros(obs_dim)
                start[:2] = start_xy
                start[2] = 0.75
                start[3] = 1.0
                goal = np.zeros(obs_dim)
                goal[:2] = goal_xy
            elif "dronemaze" in env_id.lower():
                env = DroneEnv(maze_map=maze_data,collision_checking=False)
                start_xy = env.cell_rowcol_to_xy(np.array([int(start_row), int(start_col)]))
                goal_xy = env.cell_rowcol_to_xy(np.array([int(goal_row), int(goal_col)]))
                start = np.array([start_xy[0], start_xy[1], 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                goal = np.array([goal_xy[0], goal_xy[1], 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            elif "car" in env_id.lower():
                env = CarEnv(maze_map=maze_data,collision_checking=False)
                start_xy = env.cell_rowcol_to_xy(np.array([int(start_row), int(start_col)]))
                start_rad = np.deg2rad(float(start_deg))
                goal_xy = env.cell_rowcol_to_xy(np.array([int(goal_row), int(goal_col)]))
                start = np.array([start_xy[0], start_xy[1], start_rad, 0.0, 0.0, 0.0])
                goal = np.array([goal_xy[0], goal_xy[1], 0.0, 0.0, 0.0, 0.0])
            #
            diffusion_RRT = RRT_Planner(start, goal, env_id=env_id, environment=env,
                                              sampler=diffusion_sampler_small_resnet,
                                              prediction_type=prediction_type,
                                              action_horizon=action_horizon,
                                              # edge_length=edge_length,
                                              local_map_size=local_map_size,
                                              local_map_scale=local_map_scale,
                                              global_map_scale=s_global,
                                              goal_conditioning_bias=goal_conditioning_bias,
                                              prop_duration=prop_duration,
                                              time_budget=time_budget, max_iter=300, verbose=True)
            Diffuser_MPC = MPC_Planner(start, goal, env_id=env_id, environment=env,
                                              sampler=diffusion_sampler_small_resnet,
                                              prediction_type=prediction_type,
                                              action_horizon=action_horizon,
                                              local_map_size=local_map_size,
                                              local_map_scale=local_map_scale,
                                              global_map_scale=s_global,
                                              time_budget=time_budget, verbose=True)


            # planners = [("Diffuser", Diffuser), ("diffusion_RRT", diffusion_RRT)]
            planners = [
                        # (f"diffusion_RRT_{cfg_file}", diffusion_RRT),
                        ("diffusion_RRT_PD64", diffusion_RRT),
                        # ("diffusion_RRT_PD64_fix", diffusion_RRT)
                        # ("diffusion_RRT_PD64_GB1", diffusion_RRT),
                        # ("diffusion_RRT_PD64_GB_50", diffusion_RRT),
                        # ("diffusion_RRT_PD64_GB_15", diffusion_RRT),
                        # ("diffusion_RRT_PD64_GB_85", diffusion_RRT),
                        # ("diffusion_RRT_PD64_epoch10", diffusion_RRT),
                        # ("diffusion_RRT_PDrand", diffusion_RRT),
                        # ("diffusion_RRT_PD32", diffusion_RRT),
                        # ("diffusion_RRT_PD64_DI2", diffusion_RRT),
                        # ("diffusion_RRT_PD64_DI4", diffusion_RRT),
                        # ("diffusion_RRT_PD64_DI16", diffusion_RRT),
                        # ("Diffuser_MPC", Diffuser_MPC),
                        # ("Diffuser_Random_Tree", Diffuser_random_tree),
                        # ("diffusion_RRT_drone", diffusion_RRT_drone),
                        # ("batch_diffusion_RRT_car", diffusion_RRT),
                        # ("diffusion_RRT_large_cnn", diffusion_RRT_large_cnn),
                        ]

            for planner_name, planner in planners:
                print(f"Running scenario {scenario_name} with planner {planner_name}...")
                # Prepare CSV output for each scenario and planner
                scenario_output_csv = f'benchmark_results/{scenario_name}_{planner_name}_{env_id}.csv'

                existing_rows = 0
                if os.path.exists(scenario_output_csv):
                    with open(scenario_output_csv, mode='r', newline='') as file:
                        reader = csv.reader(file)
                        rows = list(reader)
                        if len(rows) > 1:  # Exclude the header
                            existing_rows = len(rows) - 1
                remaining_runs = total_runs - existing_rows
                if remaining_runs <= 0:
                    print(f"CSV already contains {existing_rows} rows. No additional runs needed.")
                else:
                    print(f"CSV contains {existing_rows} rows. Running {remaining_runs} more iterations.")

                    # If the file is empty, create it and add headers
                    if existing_rows == 0:
                        with open(scenario_output_csv, mode='w', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(['iteration', 'success', 'runtime', 'trajectory_length', 'avg_velocity',
                                             'num_states_in_tree','num_RRT_iterations',
                                             'ctrl_effort_max', 'ctrl_effort_mean', 'ctrl_effort_std'])
                    # Run only the remaining required iterations
                    for i in range(existing_rows, existing_rows + remaining_runs):
                        print(f"Scenario: {scenario_name}, Iteration: {i + 1}/{total_runs}")
                        start_time = time.time()
                        planner.reset()
                        path_array, actions = planner.plan()
                        end_time = time.time()

                        # Collect statistics
                        planner.visualize_tree(path_array)
                        runtime = end_time - start_time
                        num_states_in_tree = planner.results["number_of_nodes"]
                        num_iterations = planner.results["iterations"]

                        if path_array is not None:
                            np.savetxt(f'path_DP_{i}.csv', path_array, fmt='%.6f', delimiter=',')
                            success = 1
                            trajectory_time = planner.results["path_time"]
                            try:
                                trajectory_length = calculate_trajectory_length(path_array)
                                avg_velocity = calculate_average_velocity(path_array)
                                ctrl_effort = np.linalg.norm(actions, axis=1)
                                ctrl_effort_max = np.max(ctrl_effort)
                                ctrl_effort_mean = np.mean(ctrl_effort)
                                ctrl_effort_std = np.std(ctrl_effort)
                            except Exception as ex:
                                success = -1
                                trajectory_length = -1
                                avg_velocity = -1
                                trajectory_time = -1
                                num_states_in_tree = -1
                                print(
                                    f"Error calculating trajectory length for scenario {scenario_name}, planner {planner_name}, run {i}")
                                print(path_array)
                                print(ex)
                                trajectory_length =0# calculate_trajectory_length(path_array)
                                avg_velocity = 0#calculate_average_velocity(path_array)
                                ctrl_effort = 0
                                ctrl_effort_max = 0
                                ctrl_effort_mean = 0
                                ctrl_effort_std = 0
                                # return
                        else:
                            success = 0
                            trajectory_length = -1
                            avg_velocity = -1
                            trajectory_time=0
                            num_states_in_tree = -1
                            ctrl_effort_max = -1
                            ctrl_effort_mean = -1
                            ctrl_effort_std = -1
                        print(f"Avg collision checking calls: {total_cc/(i+1)} for {i+1} iterations")
                        # Write results to CSV
                        with open(scenario_output_csv, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(
                                [i + 1, success, runtime, trajectory_length, trajectory_time, avg_velocity,
                                 num_states_in_tree, num_iterations, ctrl_effort_max, ctrl_effort_mean, ctrl_effort_std])




def calculate_trajectory_length(path_array):
    return np.sum(np.linalg.norm(np.diff(path_array[:, :2], axis=0), axis=1))


def calculate_average_velocity(path_array):
    velocities = np.sqrt(np.square(path_array[:, 2]) + np.square(path_array[:, 3]))
    return np.mean(velocities)


if __name__ == "__main__":

    ## Carmaze
    diffusion_sampler_checkpoints = {
        'resnet': 'carmaze.pt',
        # 'resnet': 'antmaze.pt',

    }

    # scenario_file = 'experiments/test_scenarios_ant.csv'
    scenario_file = 'experiments/test_scenarios_car.csv'

    for cfg_file in ['carmaze']: # /'antmaze'
        evaluate_all_scenarios('maps/mazes', scenario_file,  #test_scenarios.csv
                               cfg_file=cfg_file, total_runs=10, time_budget=120,
                               diffusion_sampler_checkpoints=diffusion_sampler_checkpoints)

