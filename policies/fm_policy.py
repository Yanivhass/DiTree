import os
import numpy as np
import torch
from torch import nn

from common.fm_utils import get_timesteps
from common.se3_utils import q_to_rot6d_np


class DiffusionSampler(nn.Module):
    def __init__(self, noise_pred_net, noise_scheduler, env_id,
                 policy,
                 pred_horizon,
                 action_dim,
                 prediction_type="actions",
                 obs_history=1,
                 action_history=1,
                 num_diffusion_iters=100,
                 position_conditioned=False,
                 goal_conditioned=True,
                 local_map_conditioned=True,
                 local_map_size=16,
                 # diffusion_rate=1,  # how often to sample from the diffusion model
                 ):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        metadata_path = f"metadata/{env_id}.pt"
        if os.path.exists(metadata_path):
            self.metadata = torch.load(metadata_path, weights_only=False)
        else:
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        self.action_dim = action_dim
        self.prediction_type = prediction_type

        # parameters
        self.env_id = env_id
        self.policy = policy
        self.num_diffusion_iters = num_diffusion_iters
        self.pred_horizon = pred_horizon
        self.obs_history = obs_history
        self.action_history = action_history
        self.position_conditioned = position_conditioned
        self.goal_conditioned = goal_conditioned
        self.local_map_conditioned = local_map_conditioned
        self.local_map_size = local_map_size
        # self.diffusion_rate = diffusion_rate

        self.noise_pred_net = noise_pred_net
        self.noise_scheduler = noise_scheduler

    def forward(self, obs_seq, prev_actions, goal=None, local_map=None):
        """
            obs_seq (ndarray) : (B, obs_history, obs_dim)
            prev_actions (ndarray) : (B, action_history, action_dim)
            goal (ndarray) : (obs_dim)
            local_map (ndarray) : (B, local_map_dim)
        """
        obs_seq = obs_seq.copy()    # prevent in-place modification
        if len(obs_seq.shape) == 1:
            obs_seq = np.expand_dims(obs_seq, axis=0)

        if len(obs_seq.shape) == 2:
            obs_seq = np.expand_dims(obs_seq, axis=1)

        if prev_actions is not None and len(prev_actions.shape) == 2:
            prev_actions = np.expand_dims(prev_actions, axis=0)
        batch_size = len(obs_seq)

        # normalize observation
        # curr_t = self.obs_history-1
        position = obs_seq[:, -1, :2]
        if "car" in self.env_id.lower():
            yaw = obs_seq[:, -1, 2]
            obs_seq = (obs_seq - self.metadata['Observations_mean']) / self.metadata['Observations_std']
        elif "ant" in self.env_id.lower():
            obs_seq[..., 2:] = (obs_seq[..., 2:] - self.metadata['Observations_mean']) / self.metadata['Observations_std']
            q = obs_seq[..., 3:7]
            rot = q_to_rot6d_np(q)
            obs_seq = np.concatenate([obs_seq[..., :3], rot, obs_seq[..., 7:]], axis=-1)
            yaw = np.zeros_like(obs_seq[:, -1, 0])
        elif "drone" in self.env_id.lower():
            # use 6D rotation representation for drones
            obs_seq = (obs_seq - self.metadata['Observations_mean']) / self.metadata['Observations_std']
            p = obs_seq[..., :6]
            q = obs_seq[..., 6:10]
            rot = q_to_rot6d_np(q)
            obs_seq = np.concatenate([p, rot], axis=-1)
            # yaw = np.arctan2(2 * (q[:,curr_t , 0] * q[:,curr_t, 3] + q[:,curr_t, 1] * q[:,curr_t, 2]), 1 - 2 * (q[:,curr_t, 2] ** 2 + q[:,curr_t, 3] ** 2))
            yaw = np.zeros_like(obs_seq[:, -1, 0])
        else:
            yaw = np.zeros_like(obs_seq[:, -1, 0])
            obs_seq = (obs_seq - self.metadata['Observations_mean']) / self.metadata['Observations_std']


        if self.obs_history > 0:
            obs_cond = np.zeros((batch_size, self.obs_history, obs_seq.shape[-1]))
            pad = self.obs_history - obs_seq.shape[1]
            if pad > 0:
                obs_cond[:, pad:, :] = obs_seq[:, :, :]
            else:
                obs_cond[:, :] = obs_seq[:, -self.obs_history:, :]
            obs_cond = torch.from_numpy(obs_cond)
            # reshape observation to (B,obs_history*obs_dim)

        if not self.position_conditioned:
            obs_cond = obs_cond[..., 2:] # Trim position from state
            if "car" in self.env_id.lower():
                obs_cond = obs_cond[..., 1:]  # remove yaw

        obs_cond = obs_cond.flatten(start_dim=1).to(self.device, dtype=torch.float32)
        if self.action_history > 0:
            action_cond = np.zeros((batch_size, self.action_history, self.action_dim))
            if prev_actions is not None:
                pad = self.action_history - prev_actions.shape[1]
                if pad > 0:
                    action_cond[:, pad:, :] = prev_actions[:, :, :]
                else:
                    action_cond[:, :] = prev_actions[:, -self.action_history:, :]
                action_cond = (action_cond - self.metadata['Actions_mean']) / self.metadata['Actions_std']
            action_cond = torch.from_numpy(action_cond)
            action_cond = action_cond.flatten(start_dim=1).to(self.device, dtype=torch.float32)
            obs_cond = torch.cat([obs_cond, action_cond], dim=1)
        if self.goal_conditioned:
            # use relative goal position
            curr_goal = (goal - position)
            curr_goal = torch.tensor(curr_goal).float().to(self.device)

            # rotate goal to robot frame (rotates by -yaw)
            yaw = torch.tensor(yaw, device=self.device, dtype=torch.float32)
            cos_angle = torch.cos(yaw)
            sin_angle = torch.sin(yaw)
            rotation_matrix = torch.stack([     # stacked 2x2 rotation matrices
                torch.stack([cos_angle, sin_angle], dim=1),
                torch.stack([-sin_angle, cos_angle], dim=1)
            ], dim=1)
            curr_goal = torch.matmul(rotation_matrix, curr_goal.unsqueeze(2)).squeeze(2)

            # scale goal to [-1, 1]
            scale = self.local_map_size
            curr_goal = torch.tanh(curr_goal / scale)
            obs_cond = torch.cat([obs_cond, curr_goal], dim=1)

        if self.local_map_conditioned:
            # if not tensor
            if isinstance(local_map, np.ndarray):
                local_map = torch.from_numpy(local_map)
            local_map = local_map.to(self.device, dtype=torch.float32)
            if len(local_map.shape) == 2:
                local_map = local_map.unsqueeze(0)
            local_map = local_map * 2 - 1    # scale to [-1, 1]

        # infer action
        with torch.no_grad():
            # initialize action from Guassian noise
            if self.prediction_type == "actions":
                naction = torch.randn(
                    (batch_size, self.pred_horizon, self.action_dim), device=self.device)
            elif self.prediction_type == "observations":
                naction = torch.randn(
                    (batch_size, self.pred_horizon, len(self.metadata['Observations_mean'])), device=self.device)

            if self.policy == 'diffusion':
                # init scheduler
                self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

                for k in self.noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = self.noise_pred_net(
                        sample=naction,
                        local_map=local_map,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = self.noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
            elif self.policy == 'flow_matching':
                t0, dt = get_timesteps('exp', self.num_diffusion_iters, exp_scale=4.0)
                for k in range(self.num_diffusion_iters):
                    timesteps = torch.ones((batch_size), device=self.device) * t0[k]
                    timesteps *= 20  # pos_emb_scale
                    vel_pred = self.noise_pred_net(
                        sample=naction,
                        local_map=local_map,
                        timestep=timesteps,
                        global_cond=obs_cond
                    )
                    naction = naction.detach().clone() + vel_pred * dt[k]
            else:
                raise NotImplementedError


            # unnormalize action

            if self.prediction_type == "actions":
                naction = naction.detach().to('cpu').numpy()
                naction = naction * self.metadata['Actions_std'] + self.metadata['Actions_mean']
            elif self.prediction_type == "observations":
                naction = naction * torch.tensor(self.metadata['Observations_std'],device=self.device).float() #+ self.metadata['Observations_mean']
                # rotate and translate predicted positions back to world frame
                rotation_matrix_T = rotation_matrix.transpose(1, 2)
                naction[:, :, :2] = torch.matmul(rotation_matrix_T, naction[:, :, :2].transpose(1, 2)).transpose(1, 2)
                naction[:, :, :2] = naction[:, :, :2] +  torch.tensor(position,device=self.device).float().unsqueeze(1)
                naction = naction.detach().to('cpu').numpy()

        return naction