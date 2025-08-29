import csv
import os
import pickle
import random
from datetime import datetime
import numpy as np
from tqdm.auto import tqdm
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)
import minari

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from maze_datasets import MazeDataset
from common.map_utils import load_forest, create_tree_grid
from rollout_manager import rollout

from local_map_encoder import ConditionalUnet1DWithLocalMap, ConditionalUnet1D
from log_to_tensorboard import log_results


def init_noise_pred_net(
        input_dim,
        action_dim,
        obs_dim,
        obs_history,
        action_history=0,
        goal_conditioned=True,
        goal_dim=2,
        local_map_conditioned=True,
        local_map_encoder="identity",
        local_map_embedding_dim=9,
        local_map_size=None,
        **kwargs,
):
    global_cond_dim = obs_dim * obs_history + goal_dim * goal_conditioned + action_history * action_dim
    if local_map_conditioned is not None:
        if local_map_encoder.lower() == "identity" or local_map_encoder.lower() == "mlp":
            embedding_dim = local_map_size ** 2
        else:
            embedding_dim = local_map_embedding_dim

        noise_pred_net = ConditionalUnet1DWithLocalMap(
            input_dim=input_dim,
            encoder_name=local_map_encoder,
            embedding_dim=embedding_dim,
            additional_global_cond_dim=global_cond_dim,
            local_map_size=local_map_size,
            **kwargs
        )
    else:
        noise_pred_net = ConditionalUnet1D(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            **kwargs
        )
    return noise_pred_net


def get_dataset(env_id):
    if "drone" in env_id.lower():
        if "dronemaze" in env_id.lower():
            filename = "datasets/drone_episodes.pkl"
        elif "droneforest" in env_id.lower():
            filename = "datasets/drone_forest_episodes.pkl"
        else:
            raise ValueError(f"Invalid env_id: {env_id}")
        with open(filename, "rb") as f:
            dataset = pickle.load(f)
        def get_obs(sample):
            return sample['observation']
        def get_act(sample):
            return sample['actions']

    elif "car" in env_id.lower():
        filename = "datasets/car_episodes_small.pkl"
        with open(filename, "rb") as f:
            dataset = pickle.load(f)
        def get_obs(sample):
            return sample['observation']
        def get_act(sample):
            return sample['actions']

    else:
        if isinstance(env_id, list):
            dataset = []
            for env in env_id:
                dataset.append(minari.load_dataset(env, download=False))
            dataset = minari.combine_datasets(dataset, "combined_dataset")
            env_id = env_id[0]
        else:
            dataset = minari.load_dataset(env_id, download=False)
        def get_obs(sample):
            return sample.observations['observation']
        def get_act(sample):
            return sample.actions

    # if metadata/env_id exists load, otherwise create and save
    metadata_path = f"metadata/{env_id}.pt"
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path)

        # for v_x, v_y using v range
        # metadata['Observations_min'][2] = metadata['Observations_min'][3]
        # metadata['Observations_max'][2] = metadata['Observations_max'][3]
    else:
        # Initialize arrays to store the sum and sum of squares
        Observations_sum = np.zeros_like(get_obs(dataset[0])[0])
        Observations_sum_sq = np.zeros_like(get_obs(dataset[0])[0])
        observations_min = np.min(get_obs(dataset[0]), axis=0)
        observations_max = np.max(get_obs(dataset[0]), axis=0)
        Actions_sum = np.zeros_like(get_act(dataset[0])[0])
        Actions_sum_sq = np.zeros_like(get_act(dataset[0])[0])
        actions_min = np.min(get_act(dataset[0]), axis=0)
        actions_max = np.max(get_act(dataset[0]), axis=0)

        # Iterate through each episode in the dataset
        total_steps = 0
        for episode in tqdm(dataset):
            Observations_sum += np.sum(get_obs(episode), axis=0)
            Observations_sum_sq += np.sum(get_obs(episode) ** 2, axis=0)
            observations_min = np.minimum(observations_min, np.min(get_obs(episode), axis=0))
            observations_max = np.maximum(observations_max, np.max(get_obs(episode), axis=0))
            Actions_sum += np.sum(get_act(episode), axis=0)
            Actions_sum_sq += np.sum(get_act(episode) ** 2, axis=0)
            actions_min = np.minimum(actions_min, np.min(get_act(episode), axis=0))
            actions_max = np.maximum(actions_max, np.max(get_act(episode), axis=0))
            total_steps += len(get_obs(episode))

        # Calculate mean and standard deviation
        observations_mean = Observations_sum / total_steps
        observations_std = np.sqrt(Observations_sum_sq / total_steps - observations_mean ** 2)
        actions_mean = Actions_sum / total_steps
        actions_std = np.sqrt(Actions_sum_sq / total_steps - actions_mean ** 2)

        # Pack into a dictionary
        metadata = {
            "Observations_mean": observations_mean,
            "Observations_std": observations_std,
            "Observations_min": observations_min,
            "Observations_max": observations_max,
            "Actions_mean": actions_mean,
            "Actions_std": actions_std,
            "Actions_min": actions_min,
            "Actions_max": actions_max,
        }
        os.makedirs("metadata", exist_ok=True)
        torch.save(metadata, metadata_path)

    return dataset, metadata


def train_by_steps(
        debug=False,
        # Resume training
        checkpoint=None,

        # Settings
        seed=42,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        output_dir='checkpoints/',
        experiment_name=f"diffusion_planning_{datetime.now().strftime('%d_%m_%H_%M')}",
        env_id="pointmaze-medium-v2",  # ["antmaze-large-diverse-v1", "pointmaze-medium-v2"]
        rollouts=True,
        prediction_type='observations',
        obs_history=1,  # number of observations to use per sample
        action_history=1,  # number of past actions to use per sample
        position_conditioned=False,
        goal_conditioned=True,
        goal_dim=2,
        local_map_conditioned=True,
        local_map_size=10,
        local_map_scale=0.2,
        local_map_embedding_dim=64,
        local_map_encoder="identity",
        augmentations=None,

        # Training settings
        num_epochs=100,
        batch_size=128, #256,
        num_workers=3,
        checkpoint_every=10,
        rollout_every=10,
        max_rollout_steps=250,
        num_episodes=5,  # rollout episodes

        # Diffusion settings
        policy="diffusion",  # "diffusion"/"flow_matching"
        num_diffusion_iters=100,
        unet_down_dims=[256, 512, 1024],

        # MPC settings
        pred_horizon=16,
        action_horizon=8,
):
    # set seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    s_global = 1.0
    if "antmaze" in env_id:
        env_id = 'antmaze-large-diverse-v1'
        # obs_dim = 27
        obs_dim = 29    # with 6D rotation representation
        action_dim = 8
        s_global = 4.0
        map_center = (24.0, 18.0)#(6.0, 4.5)

    elif "pointmaze" in env_id:
        env_id = "pointmaze-large-v2"
        obs_dim = 4
        if not position_conditioned:
            obs_dim -= 2  # remove (x,y)
        action_dim = 2
        if "medium" in env_id:
            map_center = (4.0, 4.0)
        elif "large" in env_id:
            map_center = (6.0, 4.5)

    elif "drone" in env_id:
        # obs_dim = 10
        obs_dim = 12    # with 6D rotation representation
        if not position_conditioned:
            obs_dim -= 2  # remove (x,y)
        action_dim = 4
        map_center = (6.0, 4.5)

    elif "car" in env_id:
        full_obs_dim = 6
        obs_dim = full_obs_dim
        if not position_conditioned:
            # obs_dim -= 2
            obs_dim -= 3 # temp no yaww
        action_dim = 2
        map_center = (6.0, 4.5)

    else:
        raise ValueError(f"Invalid env_id: {env_id}")

    os.makedirs(output_dir, exist_ok=True)
    dataset, metadata = get_dataset(env_id)


    # sizes = int(0.95 * dataset.total_episodes), int(0.05 * dataset.total_episodes)
    if debug:
        size = 32
        if "pointmaze" in env_id:
            dataset, _ = minari.split_dataset(dataset, sizes=[size, size], seed=seed)
        elif "drone" in env_id:
            dataset = dataset[:size]


    # if pointmaze or antmaze, use the maze map as global map
    if "pointmaze" in env_id:
        global_map = np.float32(dataset.env_spec.kwargs['maze_map']) if local_map_conditioned else None

    else:
        global_map = np.genfromtxt('maps/mazes/D4RL_large.csv', delimiter=',',
                                   dtype=np.float32) if local_map_conditioned else None

    train_dataset = MazeDataset(dataset, metadata,env_id, obs_history=obs_history, action_history=action_history,
                                   pred_horizon=pred_horizon, prediction_type=prediction_type,
                                   position_conditioned=position_conditioned,
                                   local_map_size=local_map_size, scale=local_map_scale,
                                   global_map=global_map, s_global=s_global, map_center=map_center,
                                   augmentations=augmentations)

    persistent_workers = True
    predict_delta = True
    if debug:
        num_workers = 0
        persistent_workers = False
        max_rollout_steps = 10
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.use('TkAgg')  # 'Qt5Agg/'TkAgg'

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )

    noise_pred_net = init_noise_pred_net(
        input_dim=action_dim if prediction_type == "actions" else full_obs_dim,
        action_dim=action_dim,
        obs_dim=obs_dim,
        obs_history=obs_history,
        action_history=action_history,
        goal_conditioned=goal_conditioned,
        goal_dim=goal_dim,
        local_map_conditioned=local_map_conditioned,
        local_map_encoder=local_map_encoder,
        local_map_embedding_dim=local_map_embedding_dim,
        local_map_size=local_map_size,
        down_dims=unet_down_dims,
    )

    noise_pred_net.to(device).train()

    noise_scheduler = None
    if policy == "diffusion":
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )

    ema = EMAModel(
        parameters=noise_pred_net.parameters(),
        power=0.75)
    ema_noise_pred_net = noise_pred_net

    optimizer = torch.optim.AdamW(
        params=noise_pred_net.parameters(),
        lr=3.0e-5, betas=(0.95, 0.999), eps=1.0e-8, weight_decay=1e-6)  #   #lr=1e-4, weight_decay=1e-6

    lr_scheduler = get_scheduler(
        name="cosine",  # 'cosine',
        optimizer=optimizer,
        num_warmup_steps=5000,    #500,
        num_training_steps=len(train_dataloader) * num_epochs
    )

    start_epoch = 1
    if checkpoint is not None:
        checkpoint = torch.load(output_dir + checkpoint)
        noise_pred_net.load_state_dict(checkpoint['noise_pred_net_state_dict'])
        ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {loss}")

    results_per_scenario, frames = rollout(
        env_id,
        policy,
        noise_pred_net,
        noise_scheduler,
        max_episode_steps=max_rollout_steps,
        num_diffusion_iters=num_diffusion_iters,
        prediction_type=prediction_type,
        obs_history=obs_history,
        action_history=action_history,
        position_conditioned=position_conditioned,
        goal_conditioned=goal_conditioned,
        local_map_size=local_map_size,
        scale=local_map_scale,
        pred_horizon=pred_horizon,
        action_horizon=action_horizon,
    )
    if frames:
        clip = ImageSequenceClip(frames, fps=10)
        os.makedirs("video", exist_ok=True)
        clip.write_videofile(f'video/{experiment_name}_epoch_{start_epoch}.mp4')

    if not debug:
        writer = SummaryWriter(log_dir=f"runs/{experiment_name}")
        log_results(writer, results_per_scenario, start_epoch - 1,
                    experiment_name, env_id,s_global)
    else:
        writer = None

    step = 0
    for epoch in range(start_epoch, num_epochs + 1):  # [1,num_epochs]
        epoch_loss = list()
        with tqdm(train_dataloader, desc=f'Epoch {epoch}') as tepoch:
            for nobs, naction, goal, local_map in tepoch:
                step += 1
                obs_cond = nobs[:, :obs_history, :]
                # device transfer
                obs_cond, naction, goal = train_dataset.normalize_samples(obs_cond, naction, goal)
                obs_cond = obs_cond.to(device).float()
                nobs = nobs.to(device).float()
                naction = naction.to(device).float()
                B = nobs.shape[0]

                # observation as FiLM conditioning
                # (B, obs_history, obs_dim)
                if not position_conditioned:
                    if "ant" not in env_id.lower():
                        obs_cond = obs_cond[..., 2:]  # remove absolute pose
                    if "car" in env_id.lower():
                        obs_cond = obs_cond[..., 1:]  # remove yaw
                # (B, obs_history * obs_dim)
                obs_cond = obs_cond.flatten(start_dim=1)
                if action_history > 0:
                    action_cond = naction[:, :action_history, :].flatten(start_dim=1)
                    # (B, obs_history * obs_dim + action_history * action_dim)
                    obs_cond = torch.cat([obs_cond, action_cond], dim=1)
                if goal_conditioned:
                    goal = goal.to(device).float()
                    # (B, obs_horizon * obs_dim + action_history * action_dim + goal_dim)
                    obs_cond = torch.cat([obs_cond, goal], dim=1)
                if local_map_conditioned:
                    local_map = local_map.to(device).float()

                # Prediction type is either "actions"(control inputs) or "observations"(states)
                if prediction_type == "actions":
                    naction = naction[:, action_history:, :]
                elif prediction_type == "observations":
                    naction = nobs[:, obs_history:, :]
                # sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)

                if policy == "diffusion":
                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()
                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)
                    # predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions, local_map, timesteps, global_cond=obs_cond)
                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)
                elif policy == "flow_matching":
                    t = torch.rand((B, 1, 1), device=device)
                    z_t = t * naction + (1.0 - t) * noise
                    target_vel = naction - noise
                    timesteps = t.squeeze() * 20  # pos_emb_scale
                    pred_vel = noise_pred_net(
                        z_t, local_map, timesteps, global_cond=obs_cond)
                    loss = nn.functional.mse_loss(pred_vel, target_vel)
                else:
                    raise NotImplementedError

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(noise_pred_net.parameters())

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)

                # save model
                if step % checkpoint_every == 0 and step > 0:
                    checkpoint = {
                        'epoch': epoch,
                        'noise_pred_net_state_dict': ema_noise_pred_net.state_dict(),
                        'ema_state_dict': ema.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                    }
                    torch.save(checkpoint, f"{output_dir}/{experiment_name}_epoch_{epoch}_step_{step}.pt")

                # rollout test
                if rollouts and ((step % rollout_every == 0 and step > 0) or debug):
                    try:
                        results_per_scenario, frames = rollout(
                            env_id,
                            policy,
                            ema_noise_pred_net,
                            noise_scheduler,
                            max_episode_steps=max_rollout_steps,
                            num_diffusion_iters=num_diffusion_iters,
                            prediction_type=prediction_type,
                            obs_history=obs_history,
                            action_history=action_history,
                            position_conditioned=position_conditioned,
                            goal_conditioned=goal_conditioned,
                            local_map_conditioned=local_map_conditioned,
                            local_map_size=local_map_size,
                            scale=local_map_scale,
                            pred_horizon=pred_horizon,
                            action_horizon=action_horizon,
                        )
                        log_results(writer, results_per_scenario, epoch,
                                    experiment_name, env_id, s_global, step=step)
                        if frames:
                            clip = ImageSequenceClip(frames, fps=10)
                            clip.write_videofile(f'video/{experiment_name}_epoch_{epoch}.mp4')
                    except Exception as e:
                        print("Rollout Failed: ")
                        print(e)

        ema_noise_pred_net = noise_pred_net
        ema.copy_to(ema_noise_pred_net.parameters())

        # log epoch loss
        if not debug:
            writer.add_scalar('Loss/train', np.mean(epoch_loss), epoch)

    # Weights of the EMA model
    # is used for inference
    ema_noise_pred_net = noise_pred_net
    ema.copy_to(ema_noise_pred_net.parameters())

    print("Done")



