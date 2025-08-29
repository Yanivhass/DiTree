import random
import time

import minari
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from scipy.spatial import KDTree

from planners.base_planner import Node, BasePlanner
from common.map_utils import create_local_map
from model.diffusion.conditional_unet1d import ConditionalUnet1D
from policies.fm_policy import DiffusionSampler


class RRT_Planner(BasePlanner):
    def __init__(self, start_state, goal_state, environment, sampler, **kwargs):
        super().__init__(start_state, goal_state, environment, sampler, **kwargs)

        self.kd_tree_dim = 2
        self.kd_tree = KDTree([start_state[:self.kd_tree_dim]])
        self.goal_sample_rate = 0.15
        self.goal_conditioning_bias = kwargs.get("goal_conditioning_bias", 0.85)  # default is 0.85
        self.prop_duration_schedule = kwargs.get("prop_duration", [64])  # default is [256,128,64] | [64]

    def reset(self):
        self.node_list.clear()
        self.node_list = [self.start_node]
        self.failed_node_list = []
        self.kd_tree = KDTree([self.start_node.state[:self.kd_tree_dim]])
        self.results = {"iterations": 0, "time": 0, "path": None, "actions": None, "number_of_nodes": 0}
        self.env.reset(options=self.options)

    def nearest_node(self, sample):
        _, index = self.kd_tree.query(sample[:, :self.kd_tree_dim], k=1)
        return self.node_list[index[0]]

    def nearest_node_batch(self, samples):
        _, index = self.kd_tree.query(samples)
        return [self.node_list[i] for i in index]

    def plan(self):
        local_map = None
        start_time = time.time()
        curr_time = time.time()
        total_diffusion_time = 0
        i = 0
        while (curr_time - start_time) < self.time_budget:
            if self.verbose:
                print(f"\rIteration: {i}, Elapsed Time: {(curr_time - start_time):.2f} seconds", end="")

            sample_node = self.random_node_sample()
            curr_node = self.nearest_node(sample_node)
            curr_state = curr_node.state
            full_action_seq = None
            full_states_seq = None
            done = False
            prev_actions = curr_node.parent_action_seq
            prev_states = curr_state[None, None, :] if curr_node.parent_states_seq is None \
                else curr_node.parent_states_seq  # (1,1,obs_dim)
            edge_length = self.prop_duration_schedule[curr_node.num_visit] if curr_node.num_visit < len(
                self.prop_duration_schedule) \
                else self.prop_duration_schedule[-1]
            # select random int value in range
            # edge_length = random.randint(self.prop_duration_schedule[0], self.prop_duration_schedule[1])
            curr_node.num_visit += 1
            # goal = self.goal_state[:2]
            # goal = sample_node[0, :2]
            if random.random() > self.goal_conditioning_bias:  # default is 0.85
                goal = sample_node[0, :2]
            else:
                goal = self.goal_state[:2]
            for j in range(edge_length // self.action_horizon):  # default is 4

                yaw = 0
                if "drone" in self.env_id.lower():
                    q = curr_state[6:10]
                    yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
                elif "car" in self.env_id.lower():
                    yaw = curr_state[2]
                local_map = create_local_map(self.maze, curr_state[0],
                                             curr_state[1], yaw,
                                             self.local_map_size,
                                             self.local_map_scale, self.s_global,
                                             (self.x_center, self.y_center))
                local_map = torch.tensor(local_map).to(self.device)
                # Sample and trim the action sequence
                start_diffusion_time = time.time()
                sampled_actions = self.sampler(prev_states, prev_actions=prev_actions,
                                               goal=goal, local_map=local_map)[0]
                end_diffusion_time = time.time()
                total_diffusion_time += (end_diffusion_time - start_diffusion_time)
                curr_state, done, curr_action_seq, curr_states_seq = self.propagate_action_sequence_env(curr_state,
                                                                                                        sampled_actions)

                # Fill edge's state/action sequence
                full_action_seq = (np.concatenate((full_action_seq, curr_action_seq))
                                   if full_action_seq is not None else curr_action_seq)
                prev_actions = curr_action_seq
                full_states_seq = (np.concatenate((full_states_seq, curr_states_seq), axis=1)
                                   if full_states_seq is not None else curr_states_seq)
                prev_states = curr_states_seq
                if done is None:  # Collision
                    if self.save_bad_edges:
                        self.failed_node_list.append(
                            Node(curr_state, full_action_seq, full_states_seq, parent=curr_node))
                    curr_state = None
                    break
                if done:
                    break

            if curr_state is not None:
                new_node = Node(curr_state, full_action_seq, full_states_seq, parent=curr_node)
                self.kd_tree = KDTree([node.state[:self.kd_tree_dim] for node in self.node_list])
                self.node_list.append(new_node)
                self.visualize_tree(filename=f'tree_plot/{i}')
            if done:
                # print diffusion time and total time
                curr_time = time.time()
                if self.verbose:
                    print(f"\rIteration: {i}, Elapsed Time: {(curr_time - start_time):.2f} seconds, "
                          f"Diffusion Time: {total_diffusion_time:.2f} seconds", end="")
                return self.handle_goal_reached(new_node, i, start_time)
                # if self.verbose:
                #     print(f" Goal reached in {i} iterations.")
                # path, actions = self.generate_final_path_env(new_node)
                # self.results["iterations"] = i
                # self.results["time"] = curr_time - start_time
                # self.results["path"] = path
                # self.results["path_time"] = len(path) * self.env_dt
                # self.results["actions"] = actions
                # self.results["number_of_nodes"] = len(self.node_list)
                # # self.visualize_tree()
                #
                # return path, actions

            i += 1
            curr_time = time.time()
        curr_time = time.time()
        if self.verbose:
            print(f"\rIteration: {i}, Elapsed Time: {(curr_time - start_time):.2f} seconds, "
                  f"Diffusion Time: {total_diffusion_time:.2f} seconds", end="")
        return self.handle_goal_not_reached(i, start_time)
        # if self.verbose:
        #     print(f" Goal not reached in {i} iterations.")
        # self.results["iterations"] = i
        # self.results["time"] = curr_time - start_time
        # self.results["number_of_nodes"] = len(self.node_list)
        # # self.visualize_tree()
        # return None, None


if __name__ == "__main__":
    # Settings
    debug = True
    time_budget = 120

    seed = 42
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env_id = 'pointmaze-medium-v2'
    checkpoint = 'pointmaze_200.pt'
    obs_horizon = 1
    num_diffusion_iters = 100
    if env_id == "antmaze-large-diverse-v1":
        obs_dim = 27
        action_dim = 8
    elif env_id == "pointmaze-medium-v2":
        obs_dim = 4
        action_dim = 2
    elif 'drone' in env_id:
        obs_dim = 10
        action_dim = 4

    dataset = minari.load_dataset(env_id, download=False)
    render_mode = 'human' if debug else 'rgb_array'
    env = dataset.recover_environment(render_mode=render_mode)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = 'checkpoints/'
    checkpoint = torch.load(output_dir + checkpoint)
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon
    )
    noise_pred_net.load_state_dict(checkpoint['model_state_dict'])
    noise_pred_net = noise_pred_net.to(device)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    episode = dataset[2]
    start = episode.observations['observation'][0]
    start[2:] = 0  # start stationary
    goal = episode.observations['desired_goal'][1]

    diffusion_sampler = DiffusionSampler(noise_pred_net, noise_scheduler, env_id,
                                         pred_horizon=16,
                                         action_dim=2,
                                         obs_history=1,
                                         goal_conditioned=False
                                         )

    diffusion_planner = RRT_planner(start, goal,
                                    env_id=env_id,
                                    environment=env,
                                    sampler=diffusion_sampler,
                                    time_budget=time_budget,
                                    max_iter=300,
                                    verbose=True,
                                    render=True,
                                    )
    print("Planning with Diffusion Sampler...")
    path_diffusion, actions_diffusion = diffusion_planner.plan()

    # kinoRRT = RRT_planner(start, goal,
    #                       env_id=env_id,  # 'pushT' or 'maze'
    #                       environment=env,
    #                       sampler=UniformSampler(env.action_space),
    #                       time_budget=time_budget,
    #                       max_iter=10000,
    #                       bounds=None  # environment bounds
    #                       )
    # print("Planning with Kino RRT...")
    # path_kino, actions_kino = kinoRRT.plan()

    # car_params = {
    #     'L': 2.0,  # Wheelbase
    #     'max_speed': 1.0  # Maximum speed
    # }

    # rrt = RRT(start, goal, car_params)
    # path = rrt.plan()

    # if path is not None:
    #     rrt.visualize(path)
    # else:
    #     print("Path not found.")
