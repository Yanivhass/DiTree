import collections
import csv
import os
import numpy as np
import torch

import gymnasium as gym
import gymnasium_robotics

from common.fm_utils import get_timesteps
from common.map_utils import create_local_map, is_colliding_parallel
from policies.fm_policy import DiffusionSampler

from drone_env import DroneEnv
from car_env import CarEnv

from policies.PD_controller import car_pd_controller

gym.register_envs(gymnasium_robotics)


def rollout(env_id, policy, ema_noise_pred_net, noise_scheduler,
                max_episode_steps=250,
                render_mode='rgb_array',
                num_diffusion_iters=100,
                prediction_type='actions',
                obs_history=1,
                action_history=1,
                position_conditioned=False,
                goal_conditioned=True,
                local_map_conditioned=True,
                local_map_size=10,  # local map will be map_size x map_size square map
                scale=0.2,  # resolution of local map
                pred_horizon=16,
                action_horizon=8,
                envs_per_scenario=32,
                render=False,
                ):
    from stable_baselines3.common.vec_env import DummyVecEnv

    render = False  # not supported for drones right now
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    s_global = 1.0
    if "antmaze" in env_id:
        env_id = 'antmaze-large-diverse-v1'
        # full_obs_dim = 27
        full_obs_dim = 29   # include position
        action_dim = 8
        s_global = 4.0

    elif "point" in env_id.lower():
        full_obs_dim = 4
        obs_dim = 4
        if not position_conditioned:
            obs_dim -= 2
        action_dim = 2
    elif "drone" in env_id.lower():
        obs_dim = 10
        full_obs_dim = 10
        if not position_conditioned:
            obs_dim -= 3  # remove (x,y,z)
        action_dim = 4
        env_class = DroneEnv
    elif "car" in env_id.lower():
        obs_dim = 6
        full_obs_dim = 6
        if not position_conditioned:
            obs_dim -= 2
        action_dim = 2
        env_class = CarEnv

    diffusion_sampler = DiffusionSampler(ema_noise_pred_net,noise_scheduler,env_id,
                                         policy,pred_horizon,action_dim,prediction_type,obs_history,
                                         action_history,num_diffusion_iters,
                                         local_map_size=local_map_size)
    diffusion_sampler = diffusion_sampler.eval().to(device)

    if "forest" in env_id.lower():
        scenarios_file = 'experiments/validation_scenarios_forest.csv'
        maps_folder = 'maps/forests'
    elif "car" in env_id.lower():
        scenarios_file = 'experiments/validation_scenarios_car.csv'
        maps_folder = 'maps/mazes'
    else:
        scenarios_file = 'experiments/validation_scenarios_maze.csv'
        maps_folder = 'maps/mazes'

    envs = []
    scenario_names = []
    mazes = []
    map_centers = []
    start_rowcol = []
    goal_rowcol = []
    start_xy = []
    goal_xy = []
    options = []
    # Load the scenarios from the CSV file

    # scenarios_file = 'experiments/debug_scenarios.csv'
    with open(scenarios_file, mode='r') as scenarios_csv:
        scenarios_reader = csv.reader(scenarios_csv)
        next(scenarios_reader)  # Skip the header
        for scenario in scenarios_reader:
            if "car" in env_id.lower():
                scenario_name, maze_name, start_row, start_col, start_deg, goal_row, goal_col = scenario
            else:
                scenario_name, maze_name, start_row, start_col, goal_row, goal_col = scenario
                start_deg = 0
            scenario_names.append(scenario_name)
            maze_path = os.path.join(maps_folder, f'{maze_name}.csv')
            if not os.path.exists(maze_path):
                print(f"Map file {maze_path} not found, skipping scenario.")
                continue
            maze = np.loadtxt(maze_path, delimiter=',')

            mazes += [maze]
            map_centers += [(s_global * np.array(maze).shape[1] / 2, s_global * np.array(maze).shape[0] / 2)]
            start_rowcol.append((int(start_row), int(start_col)))
            goal_rowcol.append((int(goal_row), int(goal_col)))
            render_mode = 'rgb_array'

            if "point" in env_id.lower():
                envs += [lambda curr_maze=maze: gym.make('PointMaze_Large-v3', maze_map=curr_maze,
                                                         render_mode=render_mode,
                                                         max_episode_steps=1000)] * envs_per_scenario
            elif "ant" in env_id.lower():
                envs += [lambda curr_maze=maze: gym.make('AntMaze_Large-v4', maze_map=curr_maze,
                                                         render_mode=render_mode,
                                                         max_episode_steps=1000)] * envs_per_scenario
            else:
                envs += [lambda curr_maze=maze: env_class(maze_map=curr_maze)] * envs_per_scenario

            options += [{
                "reset_cell": (int(start_row), int(start_col)),
                "reset_deg": float(start_deg),
                "goal_cell": (int(goal_row), int(goal_col))
            }] * envs_per_scenario

    envs = DummyVecEnv(envs)
    envs.seed(0)

    for i in range(len(mazes)):
        idx = i * envs_per_scenario
        if "point" in env_id.lower() or 'ant' in env_id.lower():
            start_xy.append(envs.envs[idx].maze.cell_rowcol_to_xy(start_rowcol[i]))
            goal_xy.append(envs.envs[idx].maze.cell_rowcol_to_xy(goal_rowcol[i]))
        else:
            start_xy.append(envs.envs[idx].cell_rowcol_to_xy(start_rowcol[i]))
            goal_xy.append(envs.envs[idx].cell_rowcol_to_xy(goal_rowcol[i]))



    metadata_path = f"metadata/{env_id}.pt"
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path)

        # for v_x, v_y using v range
        # metadata['Observations_min'][2] = metadata['Observations_min'][3]
        # metadata['Observations_max'][2] = metadata['Observations_max'][3]
    else:
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    envs.set_options(options)
    obs = envs.reset()
    if 'ant' in env_id.lower():
        obs = np.hstack((obs['achieved_goal'], obs['observation']))
    if 'point' in env_id.lower():
        obs = obs['observation']
    goal = np.array([env.goal for env in envs.envs])
    collision_count = np.zeros(envs.num_envs)
    step_to_completion = np.inf * np.ones(envs.num_envs)
    best_dist = np.inf * np.ones(envs.num_envs)
    terminated = np.zeros(envs.num_envs, dtype=bool)
    done = np.zeros(envs.num_envs, dtype=bool)
    # total_reward = 0

    frames = []
    traj = np.zeros((len(envs.envs), max_episode_steps + 1, full_obs_dim + action_dim))
    curr_step = 0
    prev_obs = collections.deque(maxlen=obs_history)
    prev_obs.append(obs)
    while curr_step < max_episode_steps and not np.all(done):

        # infer action
        with torch.no_grad():

            position = obs[:, :2]
            if "car" in env_id.lower():
                yaw = obs[:, 2]
            else:
                yaw = np.zeros_like(position[:, 0])
            local_maps = np.empty((len(position), local_map_size, local_map_size))
            if local_map_conditioned:
                for k, maze in enumerate(mazes):
                    # create local maps per maze
                    idx_start = k * envs_per_scenario
                    idx_end = (k + 1) * envs_per_scenario
                    yaw_curr = yaw[idx_start:idx_end]
                    # yaw = yaw % (2 * np.pi)
                    # yaw[:] = 0  # no rotation in local map

                    local_maps[idx_start:idx_end] = create_local_map(
                        maze,
                        position[idx_start:idx_end, 0],
                        position[idx_start:idx_end, 1],
                        yaw_curr,
                        local_map_size,
                        scale,
                        s_global,
                        map_centers[k]
                    )

                local_maps = torch.tensor(local_maps).float().to(device)
            if curr_step < action_history:
                prev_actions = None #np.zeros((len(envs.envs), action_history, action_dim))
            else:
                prev_actions = traj[:, curr_step - action_history:curr_step, full_obs_dim:]

            # prediction = diffusion_sampler(obs, prev_actions, goal, local_maps)
            prev_obs_np = np.stack(prev_obs, axis=1)
            prediction = diffusion_sampler(prev_obs_np, prev_actions, goal, local_maps)

            if prediction_type == 'actions':
                actions = prediction
            elif prediction_type == 'observations':
                actions = np.zeros((len(envs.envs),action_horizon, action_dim))

        prev_heading_error = 0
        prev_vel_error = 0
        for j in range(action_horizon):

            if render:
                frame = envs.render(mode=render_mode)  # If there are multiple environments then they are tiled together
                frames.append(frame)

            if prediction_type == 'observations':
                # not used in the paper
                target = prediction[:, int(j/5)]    # observation are at 10hz, env is at 50hz
                D_cmd, delta_cmd, prev_heading_error, prev_vel_error = car_pd_controller(obs, target, v_des=4,
                                                 prev_heading_error=prev_heading_error,prev_vel_error=prev_vel_error)
                actions[:,j,0]  = D_cmd
                actions[:,j,1]  = delta_cmd

            traj[~done, curr_step] = np.concatenate([obs[~done], actions[~done, j]], axis=-1)
            traj[done, curr_step] = traj[done, curr_step - 1]

            obs, reward, terminated, info = envs.step(actions[:, j])
            if 'ant' in env_id.lower():
                success = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'], axis=1) < 1
                position = obs['achieved_goal']
                obs = np.hstack((position, obs['observation']))
                for i, maze in enumerate(mazes):
                    start_idx = i * envs_per_scenario
                    end_idx = (i + 1) * envs_per_scenario
                    collision_count[start_idx:end_idx] += \
                        is_colliding_parallel(obs[start_idx:end_idx], maze_grid=maze,
                                              maze_size_scaling=s_global, ball_radius=1)
                terminated = np.logical_or(terminated, collision_count)
            elif 'point' in env_id.lower():
                success = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal'], axis=1) < 1
                obs = obs['observation']
                for i, maze in enumerate(mazes):
                    start_idx = i * envs_per_scenario
                    end_idx = (i + 1) * envs_per_scenario
                    collision_count[start_idx:end_idx] += \
                        is_colliding_parallel(obs[start_idx:end_idx], maze_grid=maze,
                                              maze_size_scaling=s_global, ball_radius=0.1)
                terminated = np.logical_or(terminated, collision_count)
            else:
                success = np.array([info[i]['success'] for i in range(len(info))])
                collision = np.array([info[i]['collision'] for i in range(len(info))])
                collision_count += collision
                terminated = np.logical_or(terminated, success)
            prev_obs.append(obs)

            done = np.logical_or(done, terminated)
            dist = np.linalg.norm(obs[:, :2] - goal, axis=1)
            best_dist = np.minimum(best_dist, dist)

            step_to_completion[success] = np.minimum(step_to_completion[success], curr_step)
            curr_step += 1
            print(f"Rollout Step {curr_step}/{max_episode_steps}", end='\r')
            if curr_step >= max_episode_steps or np.all(done):
                break

        if render:
            frame = envs.render()
            frames.append(frame)

    traj[:, curr_step] = np.concatenate([obs, np.zeros((obs.shape[0], action_dim))],
                                        axis=-1)
    traj[done, curr_step] = traj[done, curr_step - 1]

    step_to_completion = np.where(step_to_completion == np.inf, -1, step_to_completion)
    envs.close()

    # organize the results per environment
    results_per_scenario = []
    for i, maze in enumerate(mazes):
        start_idx = i * envs_per_scenario
        end_idx = (i + 1) * envs_per_scenario
        results_per_scenario.append({
            'scenario_name': scenario_names[i],
            'maze': maze,
            'start_rowcol': start_rowcol[i],
            'goal_rowcol': goal_rowcol[i],
            'start_position': start_xy[i],
            'goal_position': goal_xy[i],
            'best_dist': best_dist[start_idx:end_idx],
            'step_to_completion': step_to_completion[start_idx:end_idx],
            'trajectory': traj[start_idx:end_idx],
            'collision_count': collision_count[start_idx:end_idx]
        })

    return results_per_scenario, frames

# if __name__ == "__main__":
