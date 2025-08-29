import abc
import random
import time
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
import minari
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from common.map_utils import is_colliding_maze, is_colliding_drone, is_colliding_car, is_colliding_ant
from model.diffusion.conditional_unet1d import ConditionalUnet1D
from policies.fm_policy import DiffusionSampler


class Node:
    def __init__(self, state, parent_action_seq=None,parent_states_seq=None, parent=None):
        self.state = state
        self.parent_action_seq = parent_action_seq  # Action preceding this state (N,act_dim)
        self.parent_states_seq = parent_states_seq  # States preceding this state (1,N,obs_dim)
        self.parent = parent
        self.cached_actions = deque([])
        self.num_visit = 0

class BasePlanner(abc.ABC):
    def __init__(self, start_state, goal_state,
                 environment,
                 sampler,
                 action_horizon=8,
                 # edge_length=64,
                 local_map_size=(10, 10),
                 local_map_scale=0.2,
                 global_map_scale=1.0,
                 env_id='pushT',  # 'pushT'/'pointmaze'/'antmaze'/'dronemaze'
                 time_budget=10,  # time budget in seconds
                 **kwargs
                 ):
        # global_map_scale = 0.05 # changed for real world experiments
        if environment is None:
            raise ValueError("Environment is not defined.")
        self.env = environment

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if isinstance(sampler, torch.nn.Module):
            self.sampler = sampler.to(self.device)

        # Sampler settings
        self.action_horizon = action_horizon
        # self.edge_length = edge_length
        self.local_map_size = local_map_size
        self.local_map_scale = local_map_scale
        self.s_global = global_map_scale
        self.env_id = env_id

        # Planner settings
        self.time_budget = time_budget
        self.start_node = Node(start_state)
        self.goal_state = goal_state
        self.node_list = [self.start_node]
        self.results = {"iterations": 0, "time": 0, "path": None, "actions": None, "number_of_nodes": 0}

        self.render = kwargs.get('render', False)
        self.verbose = kwargs.get('verbose', False)

        # reset env
        self.env_dt = self.env.dt if hasattr(self.env, 'dt') else 0.1
        if env_id == 'pushT':
            self.env.set_state(start_state)
        elif 'point' in env_id.lower() or 'ant' in env_id.lower():
            start = self.env.maze.cell_xy_to_rowcol(start_state[:2])
            goal = self.env.maze.cell_xy_to_rowcol(goal_state[:2])
            self.options = {"reset_cell": start, "goal_cell": goal}
            self.env.reset(options=self.options)

            self.max_v = 5
            self.x_center = self.env.maze.x_map_center
            self.y_center = self.env.maze.y_map_center
            self.map_width = self.env.maze.map_width
            self.map_length = self.env.maze.map_length
            self.maze = np.float32(self.env.maze.maze_map)
        elif 'drone' in env_id.lower():
            start = self.env.cell_xy_to_rowcol(start_state[:2])
            goal = self.env.cell_xy_to_rowcol(goal_state[:2])
            self.options = {"reset_cell": start, "goal_cell": goal}
            self.env.reset(options=self.options)

            self.max_v = 5
            self.x_center = self.env.x_map_center
            self.y_center = self.env.y_map_center
            self.map_width = len(self.env.maze_map[0])
            self.map_length = len(self.env.maze_map)
            self.map_height = 1
            self.maze = np.float32(self.env.maze_map)
        elif 'car' in env_id.lower():
            start = self.env.cell_xy_to_rowcol(start_state[:2])
            goal = self.env.cell_xy_to_rowcol(goal_state[:2])
            self.options = {"reset_cell": start, "reset_deg": np.rad2deg(start_state[2]), "goal_cell": goal}
            self.env.reset(options=self.options)

            self.max_v = 5
            self.x_center = self.env.x_map_center
            self.y_center = self.env.y_map_center
            self.map_width = len(self.env.maze_map[0])
            self.map_length = len(self.env.maze_map)
            self.maze = np.float32(self.env.maze_map)

        if kwargs.get('render', False):
            self.env.render()

        # useful for debugging
        self.save_bad_edges = False
        self.failed_node_list = []

    @abc.abstractmethod
    def plan(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    def check_collision(self, state=None):
        if self.env_id == 'pushT':
            return self.env.collision
        elif 'point' in self.env_id.lower():
            return is_colliding_maze(state, self.maze)
        elif 'drone' in self.env_id.lower():
            return is_colliding_drone(state, self.maze)
        elif 'car' in self.env_id.lower():
            return is_colliding_car(state, self.maze)
        elif 'ant' in self.env_id.lower():
            return is_colliding_ant(state, self.maze,1.2,self.s_global)   # default antmaze values

    def random_node_sample(self, batch_size=1):
        if random.random() > self.goal_sample_rate:
            # get a random observation sample from the environment
            if self.env_id == 'pushT':
                return self.env.observation_space.sample()
            elif 'point' in self.env_id.lower():
                x = np.random.uniform(-self.map_width / 2, self.map_width / 2, size=(batch_size, 1))
                y = np.random.uniform(-self.map_length / 2, self.map_length / 2, size=(batch_size, 1))
                vx = np.random.uniform(-self.max_v, self.max_v, size=(batch_size, 1))
                vy = np.random.uniform(-self.max_v, self.max_v, size=(batch_size, 1))
                return np.concatenate((x, y, vx, vy), axis=1)
            elif 'drone' in self.env_id.lower():
                x = np.random.uniform(-self.map_width / 2, self.map_width / 2, size=(batch_size, 1))
                y = np.random.uniform(-self.map_length / 2, self.map_length / 2, size=(batch_size, 1))
                z = np.random.uniform(-self.map_height / 2, self.map_height / 2, size=(batch_size, 1))
                v = np.random.uniform(-self.max_v, self.max_v, size=(batch_size, 3))
                q = np.random.uniform(-1, 1, size=(batch_size, 4))
                return np.concatenate((x, y, z, v, q), axis=1)
            elif 'car' in self.env_id.lower():
                x = np.random.uniform(-self.map_width / 2, self.map_width / 2, size=(batch_size, 1))
                y = np.random.uniform(-self.map_length / 2, self.map_length / 2, size=(batch_size, 1))
                theta = np.random.uniform(-np.pi, np.pi, size=(batch_size, 1))
                v = np.random.uniform(-self.max_v, self.max_v, size=(batch_size, 1))
                throttle = np.random.uniform(-1, 1, size=(batch_size, 1))
                steer = np.random.uniform(-0.40, 0.40, size=(batch_size, 1))
                return np.concatenate((x, y, theta, v, throttle, steer), axis=1)
            elif 'ant' in self.env_id.lower():
                state = np.zeros((1,29))
                x = np.random.uniform(-self.s_global * self.map_width / 2,
                                      self.s_global * self.map_width / 2, size=(batch_size, 1))
                y = np.random.uniform(-self.s_global * self.map_length / 2,
                                      self.s_global * self.map_length / 2, size=(batch_size, 1))
                state[:,:2] = np.concatenate((x, y), axis=1)
                return state
        else:
            sample = np.zeros((batch_size, self.start_node.state.shape[0]))
            sample[:] = self.goal_state
            if 'drone' in self.env_id.lower():
                sample[:, 2] = self.map_height / 2

            return sample

    def random_node_sample_batch(self):
        if 'point' in self.env_id.lower():
            x = np.random.uniform(-self.map_width / 2, self.map_width / 2, size=(self.batch_size, 1))
            y = np.random.uniform(-self.map_length / 2, self.map_length / 2, size=(self.batch_size, 1))
            vx = np.random.uniform(-self.max_v, self.max_v, size=(self.batch_size, 1))
            vy = np.random.uniform(-self.max_v, self.max_v, size=(self.batch_size, 1))
            samples = np.concatenate((x, y, vx, vy), axis=1)
        elif 'drone' in self.env_id.lower():
            x = np.random.uniform(-self.map_width / 2, self.map_width / 2, size=(self.batch_size, 1))
            y = np.random.uniform(-self.map_length / 2, self.map_length / 2, size=(self.batch_size, 1))
            z = np.random.uniform(-self.map_height / 2, self.map_height / 2, size=(self.batch_size, 1))
            v = np.random.uniform(-self.max_v, self.max_v, size=(self.batch_size, 3))
            q = np.random.uniform(-1, 1, size=(self.batch_size, 4))
            samples = np.concatenate((x, y, z, v, q), axis=1)

        goal_indices = np.random.choice(len(x), int(self.batch_size * self.goal_sample_rate), replace=False)
        samples[goal_indices, :2] = self.goal_state[:2]
        return samples

    def dist_to_goal(self, state):
        np.linalg.norm(state[:2] - self.goal_state[:2])

    def handle_goal_reached(self, node, iterations, start_time):
        self.results["time"] = time.time() - start_time
        if self.verbose:
            print(f" Goal reached in {iterations} iterations.")
        path, actions = self.generate_final_path_env(node)
        self.results["iterations"] = iterations
        self.results["path"] = path
        self.results["path_time"] = len(path) * self.env_dt
        self.results["actions"] = actions
        self.results["number_of_nodes"] = len(self.node_list)
        # self.visualize_tree(path)
        if self.verbose:
            print(f" Goal reached in {iterations} iterations.")
        return path, actions

    def handle_goal_not_reached(self, iterations, start_time):
        self.results["time"] = time.time() - start_time
        if self.verbose:
            print(f" Goal not reached in {iterations} iterations.")
        self.results["iterations"] = iterations
        self.results["number_of_nodes"] = len(self.node_list)
        # self.visualize_tree()
        if self.verbose:
            print(f" Goal not reached in {iterations} iterations.")
        return None, None

    def propagate_action_sequence_env(self, state, action_sequence):

        if action_sequence is None:
            raise ValueError("Action sequence is None.")

        done = False
        if self.env_id == 'pushT':
            self.env.set_state(state)
        elif 'point' in self.env_id.lower():
            self.env.point_env.set_state(state[:2], state[2:])
        elif 'ant' in self.env_id.lower():
            self.env.ant_env.set_state(state[:15],state[15:])   #15,14
        else:
            self.env.set_state(state)
        states_sequence = np.zeros((self.action_horizon+1, state.shape[0]))
        states_sequence[0] = state
        check_collision_every = 4
        obs = state
        for i in range(self.action_horizon):
            # for i, action in enumerate(action_sequence):
            action = action_sequence[i]
            step_result = self.env.step(action)
            # obs, reward, done, info
            obs = step_result[0]
            if 'point' in self.env_id.lower():
                done = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']) < 0.45 * self.s_global
                obs = obs['observation']
            elif 'ant' in self.env_id.lower():
                done = np.linalg.norm(obs['achieved_goal'] - obs['desired_goal']) < 0.45 * self.s_global
                obs = np.hstack((obs['achieved_goal'], obs['observation']))
            else:
                done = step_result[4]['success']

            states_sequence[i+1] = obs
            if self.render:
                self.env.render()
            if (i % check_collision_every == 0) and (self.check_collision(obs)):
                # if self.save_bad_edges:
                #     self.failed_node_list.append(Node(obs,action_sequence[:i],states_sequence[:i]))
                # return None, done, None, None
                action_sequence = action_sequence[:i]
                states_sequence = states_sequence[:i]
                return obs, None, action_sequence, states_sequence[None,:]

            if done:
                action_sequence = action_sequence[:i]
                states_sequence = states_sequence[:i]
                break

        return obs, done, action_sequence, states_sequence[None,:]

    # def generate_final_path_env(self, final_node):
    #     actions = None
    #     path = None
    #     node = final_node
    #     while node is not None:
    #         if actions is not None and node.parent_action_seq is not None:
    #             actions = np.concatenate((node.parent_action_seq, actions))
    #         elif node.parent_action_seq is not None:
    #             actions = node.parent_action_seq
    #         # path.append(node.state)
    #         if path is not None and node.parent_states_seq is not None:
    #             path = np.concatenate((node.parent_states_seq, path))
    #         elif node.parent_states_seq is not None:
    #             path = node.parent_states_seq
    #
    #         node = node.parent
    #
    #     # path = np.float32(path)
    #     return path, actions

    def generate_final_path_env(self, final_node):
        path_list = []  # To accumulate states in order
        actions_list = []  # To accumulate actions in order

        node = final_node
        while node is not None:
            path_list = [node.state] + path_list
            if node.parent_states_seq is not None:
                path_list = list(node.parent_states_seq[0]) + path_list


            # For actions, prepend the parent's action sequence (if it exists)
            if node.parent_action_seq is not None:
                actions_list = list(node.parent_action_seq) + actions_list

            # Move to the parent node
            node = node.parent

        # Convert the lists to NumPy arrays
        path = np.array(path_list, dtype=np.float32) if path_list else None
        actions = np.array(actions_list, dtype=np.float32) if actions_list else None

        return path, actions

    def visualize(self, path=None):
        plt.figure()
        plt.grid(True)
        plt.axis("equal")

        for node in self.node_list:
            plt.plot(node.x, node.y, "go", markersize=3)

        if path is not None:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=2)

        plt.plot(self.start.x, self.start.y, "bs", markersize=10)
        plt.plot(self.goal.x, self.goal.y, "rs", markersize=10)
        plt.show()

    def visualize_tree(self, path=None,filename=None):

        # Define the custom colormap
        import matplotlib.colors as mcolors
        cmap = mcolors.ListedColormap(['#009FA6', '#E8A563'])  # 0 = blue, 1 = brown

        # Create a corresponding normalization
        bounds = [0, 0.5, 1]  # this ensures values are bucketed correctly
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        plt.figure()
        # plt.grid(True)
        plt.axis("equal")
        # plt.imshow(self.maze, cmap="binary", origin='lower',
        #            extent=[-self.x_center,self.x_center,self.y_center, -self.y_center])
        plt.imshow(self.maze, cmap=cmap, norm=norm, origin='lower',
                   extent=[-self.x_center, self.x_center, self.y_center, -self.y_center])
        # plt.scatter(self.start_node.state[0], self.start_node.state[1], color='g', s=100)
        x, y, psi = self.start_node.state[0], self.start_node.state[1], self.start_node.state[2]
        car_corners = np.array([
            [0.35 / 2, 0.2 / 2],
            [0.35 / 2, -0.2 / 2],
            [-0.35 / 2, -0.2 / 2],
            [-0.35 / 2, 0.2 / 2]
        ])
        #plot car
        car_corners = np.dot(car_corners, np.array([[np.cos(psi), -np.sin(psi)],
                                                    [np.sin(psi), np.cos(psi)]]))
        car_corners += np.array([x, y])
        plt.fill(car_corners[:, 0], car_corners[:, 1], color='g')#, alpha=0.5
        plt.scatter(self.goal_state[0], self.goal_state[1], color='r', s=100)
        plt.legend(['Start State', 'Goal'], loc='upper right')

        for node in self.node_list:
            if node.parent is not None:
                edge = node.parent_states_seq[0]   # sequence of states along tree edge
                # plt.plot(edge[:, 0], edge[:, 1], 'b-')
                plt.plot(edge[:, 0], edge[:, 1], color='#BF00FF', linewidth=1.5) ##FFA500

        if self.save_bad_edges:
            for node in self.failed_node_list:
                edge = node.parent_states_seq[0]
                # plt.plot(edge[:, 0], edge[:, 1], 'y-')
                plt.plot(edge[:, 0], edge[:, 1], color='#FFD700', linewidth=0.5, alpha=0.5) #linestyle='--'

        if path is not None:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], '#DDDDDD', linewidth=2)
            np.savetxt(f"pathMPC.csv", path, fmt='%.6f', delimiter=',') #datetime.now().strftime('%Y%m%d%H%M%S')

        # save file and date
        if filename is None:
            plt.savefig(f"tree{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
        else:
            plt.savefig(f"{filename}.png")
        plt.close()
        # plt.show()


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

    # diffusion_planner = RRT_planner(start, goal,
    #                                 env_id=env_id,
    #                                 environment=env,
    #                                 sampler=diffusion_sampler,
    #                                 time_budget=time_budget,
    #                                 max_iter=300,
    #                                 verbose=True,
    #                                 render=True,
    #                                 )
    # print("Planning with Diffusion Sampler...")
    # path_diffusion, actions_diffusion = diffusion_planner.plan()

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
