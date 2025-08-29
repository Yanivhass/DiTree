import random
import time

import minari
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from planners.base_planner import Node, BasePlanner
from common.map_utils import create_local_map
from model.diffusion.conditional_unet1d import ConditionalUnet1D
from policies.fm_policy import DiffusionSampler


class MPC_Planner(BasePlanner):
    def __init__(self, start_state, goal_state, environment, sampler, **kwargs):
        super().__init__(start_state, goal_state, environment, sampler, **kwargs)
        self.mpc_timeout = 500  # reset to start after this many iterations


    def reset(self):
        self.node_list.clear()
        self.node_list = [self.start_node]
        self.results = {"iterations": 0, "time": 0, "path": None, "actions": None, "number_of_nodes": 0}
        self.env.reset(options=self.options)

    def plan(self):
        start_time = time.time()
        goal = self.goal_state[:2]
        i = 0

        # main planning loop
        while True:
            curr_time = time.time()
            if curr_time - start_time > self.time_budget:
                break
            if self.verbose:
                print(f"\rIteration: {i}, Elapsed Time: {(curr_time - start_time):.2f} seconds", end="")
            i += 1
            curr_node = self.start_node
            curr_state = curr_node.state
            # states_sequence = curr_state[None, None, :]  # (1,1,obs_dim)
            states_sequence = curr_state[None,None,:] if curr_node.parent_states_seq is None \
                else curr_node.parent_states_seq # (1,1,obs_dim)
            prev_actions = None

            # MPC loop
            j=0
            while j < self.mpc_timeout:
                j += 1
                curr_time = time.time()
                if curr_time - start_time > self.time_budget:
                    break
                yaw = 0
                # Sample and trim the action sequence
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
                sampled_actions = self.sampler(states_sequence, prev_actions=prev_actions,
                                               goal=goal, local_map=local_map)[0]
                sampled_actions = sampled_actions[:self.action_horizon]
                curr_state, done, actions_sequence,states_sequence = self.propagate_action_sequence_env(curr_state, sampled_actions)
                prev_actions = sampled_actions


                if curr_state is not None and done is not None:
                    new_node = Node(curr_state, actions_sequence, states_sequence, parent=curr_node)

                    self.node_list.append(new_node)
                    curr_node = new_node
                    if done:
                        return self.handle_goal_reached(new_node, i, start_time)
                else:
                    break

        return self.handle_goal_not_reached(i, start_time)





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
