import random

import numpy as np
import torch
from torch.utils.data import Dataset
import bisect

from common.map_utils import is_colliding_parallel
from common.se3_utils import q_to_rot6d_th
from common.map_utils import create_local_map

class MazeDataset(Dataset):
    def __init__(self, dataset, metadata, env_id,
                 obs_history,  # number of observations in the past per sample
                 action_history,  # number of actions in the past
                 pred_horizon,
                 prediction_type='observations',  # predict either 'observations' or 'actions'
                 position_conditioned=True,  # whether to condition on the absolute position
                 global_map=None,  # whether to include local map
                 local_map_size=10,  # local map will be map_size x map_size square map
                 scale=0.2,  # resolution of local map
                 s_global=1,  # global map scaling factor
                 map_center=(0, 0),  # global map center
                 augmentations=None  # list of augmentations to apply
                 ):
        """
        Initialize the dataset.

        Args:
            dataset (list): The Minari dataset.
            pred_horizon (int): The prediction horizon (future).
            obs_history (int): The observation horizon (history).
        """

        self.env_id = env_id
        self.metadata = metadata
        self.pred_horizon = pred_horizon
        self.obs_history = obs_history
        self.action_history = action_history
        self.history_len = 1  # 1 = current state,  max(obs_history, action_history)
        self.prediction_type = prediction_type
        self.subs_rate = 1 if prediction_type == "actions" else 5  # downsample rate for actions (50->10 Hz)

        self.position_conditioned = position_conditioned
        self.global_map = global_map
        self.local_map_size = local_map_size
        self.scale = scale
        self.s_global = s_global
        self.map_center = map_center
        self.augmentations = augmentations

        # if self.prediction_type == 'observations':
        #     # downsample from 50hz to 10hz
        #     for episode in self.dataset:
        #         episode['observations'] = episode['observations'][::5]
        #         episode['actions'] = episode['actions'][::5]
        #         episode['goal'] = episode['goal'][::5]
        filtered_dataset = []
        for episode in dataset:

            if 'point' in env_id.lower():
                if len(episode.observations['observation']) > self.obs_history + self.pred_horizon:
                    filtered_dataset.append(                        {
                            'observations': episode.observations['observation'],
                            'actions': episode.actions,
                            'goal': episode.observations['desired_goal'],
                            'position': episode.observations['achieved_goal']
                        })

            elif 'ant' in env_id.lower():
                done = np.linalg.norm(episode.observations['achieved_goal'] - episode.observations['desired_goal'],
                                          axis=-1) < 0.5
                collision = is_colliding_parallel(episode.observations['achieved_goal'], global_map,
                                                      maze_size_scaling=4, ball_radius=1)
                first_invalid = np.logical_or(done, collision).argmax()
                if first_invalid > self.obs_history + self.pred_horizon:
                    filtered_dataset.append(
                        {
                            'observations': episode.observations['observation'][:first_invalid],
                            'actions': episode.actions[:first_invalid],
                            'goal': episode.observations['desired_goal'][:first_invalid],
                            'position': episode.observations['achieved_goal'][:first_invalid]
                        }
                    )

            else:
                if len(episode['observations']) / self.subs_rate > self.obs_history + self.pred_horizon:
                    filtered_dataset.append(episode)
        dataset = filtered_dataset
        self.dataset = dataset

        # Collect episodes lengths for sampling
        self.episode_samples = np.zeros((len(dataset),))
        # self.episode_samples[0] = max(len(dataset[0][self.prediction_type]) - (self.history_len - 1 + self.pred_horizon), 0)
        self.episode_samples[0] = max(len(dataset[0][self.prediction_type]) - self.pred_horizon * self.subs_rate, 0)
        # Create a cumulative length array for efficient lookup
        self.cumulative_samples = np.zeros((len(dataset),), dtype=int)
        self.cumulative_samples[0] = self.episode_samples[0]
        for i in range(1, len(dataset)):
            # self.episode_samples[i] = max(len(dataset[i][self.prediction_type]) - (self.history_len - 1 + self.pred_horizon), 0)
            self.episode_samples[i] = max(len(dataset[i][self.prediction_type]) - self.pred_horizon * self.subs_rate, 0)
            self.cumulative_samples[i] = self.cumulative_samples[i - 1] + self.episode_samples[i]
        # self.cumulative_samples = np.cumsum(self.episode_samples)

        self.Observations_mean = metadata['Observations_mean']
        self.Observations_std = metadata['Observations_std']
        # self.Observations_min = metadata['Observations_min']
        # self.Observations_max = metadata['Observations_max']
        # if not self.position_conditioned:
        #     self.Observations_mean = metadata['Observations_mean'][3:]
        #     self.Observations_std = metadata['Observations_std'][3:]
        #     self.Observations_min = metadata['Observations_min'][3:]
        #     self.Observations_max = metadata['Observations_max'][3:]
        self.Actions_mean = metadata['Actions_mean']
        self.Actions_std = metadata['Actions_std']
        # self.Actions_min = metadata['Actions_min']
        # self.Actions_max = metadata['Actions_max']

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.cumulative_samples[-1]

    def normalize_samples(self, obs_seq, act_seq, goal):
        """
        Normalize the observation and action sequences.

        Args:
            obs_seq (np.ndarray): A sequence of observations.
            act_seq (np.ndarray): A sequence of actions.

        Returns:
            obs_seq (np.ndarray): Normalized observation sequence.
            act_seq (np.ndarray): Normalized action sequence.
        """
        if "car" in self.env_id.lower():
            obs_seq[:, :, 2] = (obs_seq[:, :, 2] + np.pi) % (2 * np.pi) - np.pi  # normalize angle to [-pi, pi]
        elif "drone" in self.env_id.lower():
            obs_seq[..., :10] = (obs_seq[..., :10] - self.Observations_mean) / self.Observations_std
        elif "ant" in self.env_id.lower():
            obs_seq[...,0] = (obs_seq[...,0] - self.Observations_mean[0]) / self.Observations_std[0]
            obs_seq[...,7:] = (obs_seq[...,7:] - self.Observations_mean[...,5:]) / self.Observations_std[...,5:]
        else:
            obs_seq = (obs_seq - self.Observations_mean) / self.Observations_std
        # obs_seq = (obs_seq - self.Observations_min) / (self.Observations_max - self.Observations_min)
        # obs_seq = obs_seq * 2 - 1

        act_seq = (act_seq - self.Actions_mean) / self.Actions_std
        # act_seq = (act_seq - self.Actions_min) / (self.Actions_max - self.Actions_min)
        # act_seq = act_seq * 2 - 1
        # if self.position_conditioned:
        #     goal = (goal - self.metadata['Observations_mean'][0:2])
        # goal /= self.metadata['Observations_std'][0:2]

        return obs_seq, act_seq, goal

    def denormalize_samples(self, obs_seq, act_seq, goal):
        """
        Denormalize the observation and action sequences.

        Args:
            obs_seq (np.ndarray): A sequence of normalized observations.
            act_seq (np.ndarray): A sequence of normalized actions.

        Returns:
            obs_seq (np.ndarray): Denormalized observation sequence.
            act_seq (np.ndarray): Denormalized action sequence.
        """
        # obs_seq = obs_seq * self.Observations_std + self.Observations_mean
        # act_seq = act_seq * self.Actions_std + self.Actions_mean
        obs_seq = (obs_seq + 1) / 2
        obs_seq = obs_seq * (self.Observations_max - self.Observations_min) + self.Observations_min
        act_seq = (act_seq + 1) / 2
        act_seq = act_seq * (self.Actions_max - self.Actions_min) + self.Actions_min
        # goal = goal * self.metadata['Observations_std'][0:2]
        # if self.position_conditioned:
        #     goal += self.metadata['Observations_mean'][0:2]

        return obs_seq, act_seq, goal

    def normalize_goal(self, goal):
        """
        Normalize the goal.

        Args:
            goal (np.ndarray): The goal to normalize.

        Returns:
            goal (np.ndarray): Normalized goal.
        """
        return (goal - self.metadata['Observations_mean'][0:2]) / self.metadata['Observations_std'][0:2]

    def denormalize_goal(self, goal):
        """
        Denormalize the goal.

        Args:
            goal (np.ndarray): The goal to denormalize.

        Returns:
            goal (np.ndarray): Denormalized goal.
        """
        return goal * self.metadata['Observations_std'][0:2] + self.metadata['Observations_mean'][0:2]

    def sample(self):
        """
        Sample a random timestep t at a random episode and return the observation and action sequences.

        Returns:
            obs_seq (np.ndarray): A sequence of observations from t-obs_horizon to t.
            act_seq (np.ndarray): A sequence of actions from t to t+pred_horizon.
        """
        # Sample a random episode index
        episode_index = np.random.randint(0, len(self.dataset.episodes))
        episode = self.dataset.episodes[episode_index]

        # Calculate valid range for t to ensure we don't go out of bounds
        max_t = len(episode.observations) - self.pred_horizon
        min_t = self.obs_horizon

        # Sample a random timestep t within the valid range
        t = np.random.randint(min_t, max_t)

        # Extract observation and action sequences
        obs_seq = episode.observations[t - self.obs_horizon: t]
        act_seq = episode.actions[t: t + self.pred_horizon]

        return obs_seq, act_seq

    def apply_augmentations(self, obs_seq, act_seq, goal, local_map):
        """
        Apply augmentations to the sequences and local map.

        Args:
            obs_seq (torch.Tensor): Observation sequence.
            act_seq (torch.Tensor): Action sequence.
            goal (torch.Tensor): Goal.
            local_map (torch.Tensor): Local map.

        Returns:
            Augmented obs_seq, act_seq, goal, and local_map.
        """

        if "car" in self.env_id.lower():
            for aug in self.augmentations:
                # if aug == 'rotate':
                #     angle = random.choice([0, 90, 180, 270])
                #     if angle != 0:
                #         local_map = torch.rot90(local_map, k=angle // 90)
                #         cos_angle = np.cos(np.radians(angle))
                #         sin_angle = np.sin(np.radians(angle))
                #         rotation_matrix = torch.tensor([[cos_angle, -sin_angle], [sin_angle, cos_angle]],
                #                                        dtype=torch.float32)
                #         goal = torch.matmul(rotation_matrix, goal)
                #         obs_seq[:, :2] = torch.matmul(obs_seq, rotation_matrix)
                #         act_seq[:, :2] = torch.matmul(act_seq[:, :2], rotation_matrix)
                if aug == 'mirror':
                    if random.random() > 0.5:
                            local_map = torch.flip(local_map, [0])
                            goal[1] = -goal[1]
                            obs_seq[:, 2] = -obs_seq[:, 2]  # heading (yaw) [rad]
                            obs_seq[:, 5] = -obs_seq[:, 5]  # steering angle [rad]
                            act_seq[:, 1] = -act_seq[:, 1]  # time derivative of the steering angle [rad]
        if "point" in self.env_id.lower():
            for aug in self.augmentations:
                if aug == 'rotate':
                    angle = random.choice([0, 90, 180, 270])
                    if angle != 0:
                        local_map = torch.rot90(local_map, k=angle // 90)
                        cos_angle = np.cos(np.radians(angle))
                        sin_angle = np.sin(np.radians(angle))
                        rotation_matrix = torch.tensor([[cos_angle, -sin_angle], [sin_angle, cos_angle]],
                                                       dtype=torch.float32)
                        goal = torch.matmul(rotation_matrix, goal)
                        obs_seq[:, :2] = torch.matmul(obs_seq, rotation_matrix)
                        act_seq[:, :2] = torch.matmul(act_seq[:, :2], rotation_matrix)
                elif aug == 'mirror':
                    if random.random() > 0.5:
                        local_map = torch.flip(local_map, [1])
                        goal[0] = -goal[0]
                        obs_seq[:, 0] = -obs_seq[:, 0]
                        act_seq[:, 0] = -act_seq[:, 0]
                    if random.random() > 0.5:
                        local_map = torch.flip(local_map, [0])
                        goal[1] = -goal[1]
                        obs_seq[:, 1] = -obs_seq[:, 1]
                        act_seq[:, 1] = -act_seq[:, 1]

        return obs_seq, act_seq, goal, local_map

    def __getitem__(self, idx):
        """
        Sample a random timestep t at a random episode and return the observation and action sequences.

        Args:
            idx (int): Index

        Returns:
            obs_seq (torch.Tensor): A sequence of observations from t-obs_history to t.
            act_seq (torch.Tensor): A sequence of actions from t to t+pred_horizon.
        """
        # Use binary search to find the episode index
        episode_index = bisect.bisect_right(self.cumulative_samples, idx)
        episode = self.dataset[episode_index]
        i = idx
        if episode_index > 0:
            i = idx - self.cumulative_samples[episode_index - 1]

        t = i  # + self.obs_history
        T = len(episode[self.prediction_type])

        # Handle action sequence
        if t + self.pred_horizon > T:
            print(f"idx={idx} t={t} T={T}")

        if self.prediction_type == 'actions':
            obs_pred_horizon = 0
            act_pred_horizon = self.pred_horizon
        else:
            obs_pred_horizon = self.pred_horizon
            act_pred_horizon = 0

        if t < (self.obs_history - 1) * self.subs_rate:
            # fill with zeros
            obs_padding = np.zeros((((self.obs_history - 1) * self.subs_rate - t) // self.subs_rate,
                                    self.metadata['Observations_mean'].shape[0]))
            obs_seq = np.vstack(
                [obs_padding, episode['observations'][:t + (obs_pred_horizon + 1) * self.subs_rate:self.subs_rate]])
        else:
            obs_seq = episode['observations'][(t - (self.obs_history - 1) * self.subs_rate):(
                        t + (obs_pred_horizon + 1) * self.subs_rate):self.subs_rate]
        if t < self.action_history:
            # fill with zeros
            act_padding = np.zeros((self.action_history - t, self.metadata['Actions_mean'].shape[0]))
            act_seq = np.vstack(
                [act_padding, episode['actions'][:(t + act_pred_horizon * self.subs_rate):self.subs_rate]])
        else:
            act_seq = episode['actions'][
                      (t - self.action_history):(t + act_pred_horizon * self.subs_rate):self.subs_rate]
        # if t - self.action_history >= 0:
        #     act_seq = episode['actions'][(t - self.action_history):(t + self.pred_horizon)]
        # else:
        #     # fill with mean action
        #     act_seq = np.vstack([[self.Actions_mean] * (self.action_history - t), episode['actions'][:t + self.pred_horizon]])

        goal = episode['goal']
        if "ant" in self.env_id.lower() or "point" in self.env_id.lower():
            goal = goal[t]
        else:
            x = (goal[1] + 0.5) * self.s_global - self.map_center[0]
            y = self.map_center[1] - (goal[0] + 0.5) * self.s_global
            goal = np.array([x, y])

        # Convert to PyTorch tensors
        obs_seq = torch.tensor(obs_seq, dtype=torch.float32)
        act_seq = torch.tensor(act_seq, dtype=torch.float32)
        goal = torch.tensor(goal, dtype=torch.float32)

        if (obs_seq.shape[0] != self.obs_history + obs_pred_horizon or
                act_seq.shape[0] != self.action_history + act_pred_horizon):
            print("dataset getitem error:")
            print("obs_seq", obs_seq.shape)
            print("act_seq", act_seq.shape)

        local_map = []
        if self.global_map is not None:
            # create local raster map\occupancy grid
            if "ant" in self.env_id.lower():
                x,y = episode['position'][t]
            else:
                x, y = obs_seq[self.obs_history - 1, 0].item(), obs_seq[self.obs_history - 1, 1].item()

            if "car" in self.env_id.lower():
                yaw = obs_seq[self.obs_history - 1, 2].item()
            else:
                yaw = 0
            local_map = create_local_map(self.global_map, x, y, yaw, self.local_map_size,
                                         self.scale, self.s_global, self.map_center)
            local_map = torch.tensor(local_map, dtype=torch.float32).squeeze(0)
            # scale to [-1, 1]
            local_map = local_map * 2 - 1

        if "car" in self.env_id.lower():
            # rotate to car frame
            cos_angle = np.cos(yaw)
            sin_angle = np.sin(yaw)
            rotation_matrix = torch.tensor([  # rotates by -yaw
                [cos_angle, sin_angle],
                [-sin_angle, cos_angle]
            ], dtype=torch.float32)
            goal = goal - obs_seq[self.obs_history - 1, :2]  # goal relative to the current position
            goal = torch.matmul(rotation_matrix, goal)
            goal = np.tanh(goal / self.local_map_size)  # scale goal to [-1, 1]

            # obs_seq = obs_seq[:, 3:]    # remove x, y, theta for car
            # future observations are relative to current state
            obs_seq[self.obs_history:] = obs_seq[self.obs_history:] - obs_seq[self.obs_history - 1]
            obs_seq[self.obs_history:, :2] = torch.matmul(rotation_matrix,
                                                          obs_seq[self.obs_history:, :2].t()).t()  # rotate to car frame
            obs_seq[:, 2] = torch.arctan2(torch.sin(obs_seq[:, 2]),
                                          torch.cos(obs_seq[:, 2]))  # normalize angle to [-pi, pi]
            obs_seq[self.obs_history:] /= self.metadata['Observations_std']  # normalize
        else:
            # goal relative to the current position
            if "ant" in self.env_id.lower():
                goal = goal - episode['position'][t]
            else:
                goal = goal - obs_seq[self.obs_history - 1, :2]
            goal = np.tanh(goal / self.local_map_size)  # scale goal to [-1, 1]

        # Apply augmentations
        if self.augmentations is not None:
            obs_seq, act_seq, goal, local_map = self.apply_augmentations(obs_seq, act_seq, goal, local_map)

        # use 6D rotation representation
        if "drone" in self.env_id.lower():
            pos = obs_seq[..., :6]
            q = obs_seq[..., 6:10]
            rot = q_to_rot6d_th(q)
            obs_seq = torch.cat([pos, rot], dim=-1)
        if "ant" in self.env_id.lower():
            q = obs_seq[..., 1:5]
            rot = q_to_rot6d_th(q)
            obs_seq = np.concatenate([obs_seq[..., :1], rot, obs_seq[..., 5:]], axis=-1)

        return obs_seq, act_seq, goal, local_map
