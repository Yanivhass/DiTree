from matplotlib.collections import LineCollection
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import io


def log_results(writer: SummaryWriter, results_per_scenario, epoch,experiment_name, env_id="drone", s_global=1.0, save_to_pkl=False, pkl_path=None,step=0):
    for scenario_idx, result in enumerate(results_per_scenario):
        scenario_name = result['scenario_name']
        maze = result['maze']
        start_position = result['start_position']
        goal_position = result['goal_position']
        map_center = [s_global*len(maze[0])/2, s_global*len(maze)/2]

        best_dist = result['best_dist']
        step_to_completion = result['step_to_completion']
        collision_count = result['collision_count']
        trajectories = result['trajectory']
        if "drone" in env_id.lower():
            avg_velocity = np.sqrt((trajectories[:, :, 3:6] ** 2).sum(axis=-1)).mean(axis=1)
            avg_height = trajectories[:, :, 2].mean(axis=1)
        else:
            avg_velocity = np.sqrt(trajectories[:, :, 2] ** 2 + trajectories[:, :, 3] ** 2).mean(axis=1)
        if writer is not None:
            writer.add_histogram(f'Scenario_{scenario_name}/Best_Distance', best_dist, epoch)
            writer.add_scalar(f'Scenario_{scenario_name}/Mean_Best_Distance', np.mean(best_dist), epoch)
            writer.add_scalar(f'Scenario_{scenario_name}/Std_Best_Distance', np.std(best_dist), epoch)
            if "drone" in env_id.lower():
                writer.add_histogram(f'Scenario_{scenario_name}/Avg_Height', avg_height, epoch)
                writer.add_scalar(f'Scenario_{scenario_name}/Mean_Avg_Height', np.mean(avg_height), epoch)
                writer.add_scalar(f'Scenario_{scenario_name}/Std_Avg_Height', np.std(avg_height), epoch)

            valid_step_to_completion = step_to_completion[step_to_completion > 0]
            if len(valid_step_to_completion) > 0:
                writer.add_histogram(f'Scenario_{scenario_name}/Step_To_Completion',
                                     step_to_completion[step_to_completion > 0], epoch)
                writer.add_scalar(f'Scenario_{scenario_name}/Mean_Step_To_Completion',
                                  np.mean(step_to_completion[step_to_completion > 0]),
                                  epoch)
                writer.add_scalar(f'Scenario_{scenario_name}/Std_Step_To_Completion',
                                  np.std(step_to_completion[step_to_completion > 0]),
                                  epoch)

            writer.add_histogram(f'Scenario_{scenario_name}/Collision_Count', collision_count, epoch)
            writer.add_scalar(f'Scenario_{scenario_name}/Mean_Collision_Count', np.mean(collision_count), epoch)
            writer.add_scalar(f'Scenario_{scenario_name}/Std_Collision_Count', np.std(collision_count), epoch)

            writer.add_histogram(f'Scenario_{scenario_name}/Avg_Velocities', avg_velocity, epoch)
            writer.add_scalar(f'Scenario_{scenario_name}/Mean_Velocities', np.mean(avg_velocity), epoch)
            writer.add_scalar(f'Scenario_{scenario_name}/Std_Velocities', np.std(avg_velocity), epoch)

        # Plot and save trajectories for each scenario
        plt.figure(figsize=(10, 10))
        plt.imshow(maze, cmap="binary", origin='lower',
                   extent=(-map_center[0], map_center[0],
                           map_center[1], -map_center[1]))
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title(f"Trajectories for Scenario {scenario_name} epoch {epoch}")
        ax = plt.gca()


        for trajectory in trajectories:
            # Identify rows that are not entirely zeros
            non_zero_rows = np.any(trajectory != 0, axis=1)
            trajectory = trajectory[non_zero_rows]
            if "drone" in env_id.lower():
                x, y, height = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

                # Create line segments for the trajectory
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Create a LineCollection with color mapped to height
                lc = LineCollection(segments, cmap='viridis', linewidth=2)
                lc.set_array(height)  # Set height as the color array
                lc.set_clim(0, 1)
                ax.add_collection(lc)
            else:
                if trajectory.size > 0:
                    plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.5)

        if "drone" in env_id.lower():
            cbar = plt.colorbar(lc, label='Height')
            # cbar.set_clim(0, 1)

        # Plot start and goal positions
        # start_x = (start_position[1] + 0.5) - map_center[0]
        # start_y = map_center[1] - (start_position[0] + 0.5)
        start_x = start_position[0]
        start_y = start_position[1]
        # goal_x = (goal_position[1] + 0.5) - map_center[0]
        # goal_y = map_center[1] - (goal_position[0] + 0.5)
        goal_x = goal_position[0]
        goal_y = goal_position[1]
        plt.scatter(start_x, start_y, color='green', marker='o', s=100, label='Start')
        plt.scatter(goal_x, goal_y, color='red', marker='x', s=100, label='Goal')
        plt.legend()

        # Save plot to TensorBoard
        if writer is not None:
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = np.array(plt.imread(buf))
            writer.add_image(f'Scenario_{scenario_name}/Trajectories', img, epoch, dataformats='HWC')

       # Save plot locally
        local_save_path = f'plots/{experiment_name}/scenario_{scenario_name}_epoch_{epoch}_step_{step}.png'
        os.makedirs(os.path.dirname(local_save_path), exist_ok=True)
        plt.savefig(local_save_path)

        plt.close()

        if save_to_pkl:
            if not os.path.exists(pkl_path):
                os.makedirs(pkl_path)
            with open(os.path.join(pkl_path, f'trajectories_scenario_{scenario_name}_epoch_{epoch}.pkl'), 'wb') as f:
                torch.save(trajectories, f)
