import csv
import math
import numpy as np
from PIL import Image

from ManiCM.diffusion_tree.common.se3_utils import q_to_rot_mat_np




def print_calls():
    global cc_calls
    print(cc_calls)

def is_colliding_car(state, maze_map, ball_radius=0.1, car_length=0.15):
    offset = (car_length * 0.5) * np.array([np.cos(state[2]), np.sin(state[2])])
    ball_centers = np.array([
        state[:2] + offset,  # Front ball
        state[:2] - offset  # Rear ball
    ])
    collision = is_colliding_parallel(ball_centers,maze_map,ball_radius=ball_radius)
    return collision.any()


def is_colliding_drone(state, maze_map, drone_radius=0.1):
    if state[2] < drone_radius or state[2] > 1 - drone_radius:
        collision = True
    else:
        collision = is_colliding_maze(state[0:3], maze_map)
    return collision


def is_colliding_ant(state, maze_map, ant_radius=1,map_scale=1):
    rotation_matrix = q_to_rot_mat_np(state[3:7])   # antmaze uses [x,y,z,w]
    z_body = rotation_matrix[:, 2]  # Get the z-axis of the body frame
    cos_theta = z_body[2]  # Cosine of the angle between the z-axis and the global z-axis, equivalent to dot(z_body, [0, 0, 1])
    if cos_theta < 0:
        # ant is upside down
        collision = True
    else:
        collision = is_colliding_maze(state[0:3], maze_map,map_scale, ant_radius)
    return collision


def is_colliding_maze(state, maze_grid,maze_size_scaling=1, ball_radius=0.1):
    """
    Check if the ball with radius collides with any walls in the maze, including side and diagonal checks.

    Args:
        state (list): The state of the ball, where the first two elements are the x and y coordinates.
        ball_radius (float): The radius of the ball.
        env (object): The environment object containing the maze.

    Returns:
        bool: True if there is a collision with a wall, False otherwise.
    """
    agent_x, agent_y = state[:2]

    map_length = maze_grid.shape[0]
    map_width = maze_grid.shape[1]
    x_map_center = map_width / 2 * maze_size_scaling
    y_map_center = map_length / 2 * maze_size_scaling

    row = np.floor((y_map_center - agent_y) / maze_size_scaling).astype(int)
    col = np.floor((agent_x + x_map_center) / maze_size_scaling).astype(int)

    cell_x = (col + 0.5) * maze_size_scaling - x_map_center
    cell_y = y_map_center - (row + 0.5) * maze_size_scaling

    # Get the current cell's bounds
    current_cell_x_min = cell_x - maze_size_scaling / 2
    current_cell_x_max = cell_x + maze_size_scaling / 2
    current_cell_y_min = cell_y - maze_size_scaling / 2
    current_cell_y_max = cell_y + maze_size_scaling / 2

    # Ensure that row and col are within bounds
    if not (0 <= row < len(maze_grid)) or not (0 <= col < len(maze_grid[0])):
        return True  # Ball is out of maze bounds

    # Right side check (checking the right side wall of the current cell)
    agent_pos_right = agent_x + ball_radius
    if agent_pos_right > current_cell_x_max:
        check_row, check_col = row, col + 1  # Checking the right cell
        if (check_col >= len(maze_grid[0]) or maze_grid[row][col + 1] == 1):
            return True  # Collision with the right side wall

    # Left side check (checking the left side wall of the current cell)
    agent_pos_left = agent_x - ball_radius
    if agent_pos_left < current_cell_x_min:
        check_row, check_col = row, col - 1  # Checking the left cell
        if (check_col < 0 or maze_grid[check_row][check_col] == 1):
            return True  # Collision with the left side wall

    # Top side check (checking the top side wall of the current cell)
    agent_pos_top = agent_y + ball_radius
    if agent_pos_top > current_cell_y_max:
        check_row, check_col = row - 1, col  # Checking the top cell
        if (check_row < 0 or maze_grid[check_row][check_col] == 1):
            return True  # Collision with the top side wall

    # Bottom side check (checking the bottom side wall of the current cell)
    agent_pos_bottom = agent_y - ball_radius
    if agent_pos_bottom < current_cell_y_min:
        check_row, check_col = row + 1, col  # Checking the bottom cell
        if (check_row >= len(maze_grid) or maze_grid[check_row][check_col] == 1):
            return True  # Collision with the bottom side wall

    # Diagonal checks: check for collisions in the corners of the current cell
    corners = [
        (current_cell_x_max, current_cell_y_max, row - 1, col + 1),  # Top-right corner
        (current_cell_x_min, current_cell_y_max, row - 1, col - 1),  # Top-left corner
        (current_cell_x_max, current_cell_y_min, row + 1, col + 1),  # Bottom-right corner
        (current_cell_x_min, current_cell_y_min, row + 1, col - 1)  # Bottom-left corner
    ]

    for corner_x, corner_y, check_row, check_col in corners:
        dist_to_corner = math.sqrt((corner_x - agent_x) ** 2 + (corner_y - agent_y) ** 2)
        if dist_to_corner < ball_radius:
            # Ensure check_row and check_col are within bounds
            if 0 <= check_row < len(maze_grid) and 0 <= check_col < len(maze_grid[0]):
                if maze_grid[check_row][check_col] == 1:
                    return True  # Collision with the diagonal wall

    return False  # No collision detected


def is_colliding_parallel(states, maze_grid, maze_size_scaling=1, ball_radius=0.1):
    """
    Optimized vectorized collision detection for multiple states in a maze.

    Args:
        states (np.ndarray): An array of shape (N, 2), where N is the number of states,
                             and the first two elements in each state are the x and y coordinates.
        maze_grid (np.ndarray): A 2D NumPy array representing the maze grid.
                                0 for empty cells, 1 for walls.
        maze_size_scaling (float): The size scaling of the maze (size of each cell).
        ball_radius (float): The radius of the ball.

    Returns:
        np.ndarray: A boolean array of shape (N,), where True indicates a collision.
    """

    # maze_size_scaling = 0.05 # changed for real world experiments
    if len(states.shape) == 1:
        states = np.expand_dims(states, axis=0)
    N = states.shape[0]
    agent_x = states[:, 0]
    agent_y = states[:, 1]

    # Compute map dimensions and centers
    map_length = maze_grid.shape[0]
    map_width = maze_grid.shape[1]
    x_map_center = map_width / 2 * maze_size_scaling
    y_map_center = map_length / 2 * maze_size_scaling

    # Compute cell indices for each agent using the maze's coordinate transformations
    rows = np.floor((y_map_center - agent_y) / maze_size_scaling).astype(int)
    cols = np.floor((agent_x + x_map_center) / maze_size_scaling).astype(int)

    # Check if outside the maze bounds
    out_of_bounds_idx = (rows < 0) | (rows >= map_length) | (cols < 0) | (cols >= map_width)
    collision = out_of_bounds_idx

    # Check if inside a wall
    inside_wall_idx = maze_grid[rows, cols] == 1

    collision = collision | inside_wall_idx

    if np.all(collision):
        return collision

    # Compute the center positions of the current cells
    cell_x = (cols + 0.5) * maze_size_scaling - x_map_center
    cell_y = y_map_center - (rows + 0.5) * maze_size_scaling

    # Current cell bounds
    half_cell = maze_size_scaling / 2
    current_cell_x_min = cell_x - half_cell
    current_cell_x_max = cell_x + half_cell
    current_cell_y_min = cell_y - half_cell
    current_cell_y_max = cell_y + half_cell

    # Precompute agent positions plus/minus radius
    agent_pos_right = agent_x + ball_radius
    agent_pos_left = agent_x - ball_radius
    agent_pos_top = agent_y + ball_radius
    agent_pos_bottom = agent_y - ball_radius

    # Side checks
    # Right side
    check_i = rows
    check_j = cols + 1
    check_j = np.clip(check_j, 0, map_width - 1)
    collision = collision | (agent_pos_right > current_cell_x_max) & (maze_grid[check_i, check_j] == 1)

    # Left side
    check_i = rows
    check_j = cols - 1
    check_j = np.clip(check_j, 0, map_width - 1)
    collision = collision | (agent_pos_left < current_cell_x_min) & (maze_grid[check_i, check_j] == 1)

    # Top side
    check_i = rows - 1
    check_i = np.clip(check_i, 0, map_length - 1)
    check_j = cols
    collision = collision | (agent_pos_top > current_cell_y_max) & (maze_grid[check_i, check_j] == 1)

    # Bottom side
    check_i = rows + 1
    check_i = np.clip(check_i, 0, map_length - 1)
    check_j = cols
    collision = collision | (agent_pos_bottom < current_cell_y_min) & (maze_grid[check_i, check_j] == 1)

    if np.all(collision):
        return collision

    # Diagonal checks
    corners = [
        (current_cell_x_max, current_cell_y_max, rows - 1, cols + 1),  # Top-right
        (current_cell_x_min, current_cell_y_max, rows - 1, cols - 1),  # Top-left
        (current_cell_x_max, current_cell_y_min, rows + 1, cols + 1),  # Bottom-right
        (current_cell_x_min, current_cell_y_min, rows + 1, cols - 1)  # Bottom-left
    ]

    for corner_x, corner_y, check_i, check_j in corners:
        dist_to_corner = np.hypot(corner_x - agent_x, corner_y - agent_y)
        invalid_cell = (check_i < 0) | (check_i >= map_length) | (check_j < 0) | (check_j >= map_width)
        check_i = np.clip(check_i,0,map_length-1)
        check_j = np.clip(check_j,0,map_width-1)
        collision = collision | invalid_cell | (dist_to_corner < ball_radius) & ((maze_grid[check_i, check_j] == 1))

    return collision


def create_local_map(global_map, x, y, theta, map_size, scale, s_global, map_center):
    '''
    Creates a local occupancy map from the robot's perspective.

    Parameters:
    - global_map: 2D numpy array representing the global occupancy grid map.
    - x, y: Robot's position in global coordinates (meters).
    - theta: Robot's heading (radians), where 0 means facing the positive x-direction.
    - map_size: Tuple (N, N), desired size of the local map in pixels.
    - scale: Resolution of the local map (meters per pixel).
    - s_global: Resolution of the global map (meters per pixel).

    Returns:
    - local_map: 2D numpy array representing the local occupancy map.
    '''

    # if x is scalar, convert to numpy array
    if isinstance(x, (int, float, np.generic)):
        x = np.array([x])
        y = np.array([y])
        theta = np.array([theta])

    K = len(x)  # Number of robots (batch size)

    # Determine the size of the local map
    if isinstance(map_size, (int, float)):
        N = int(map_size)
    else:
        N = int(map_size[0])  # Size of the local map (N x N)
    L = N * scale  # Physical length of the local map (meters)

    # Create local grid coordinates centered around the robot
    xs = np.linspace(-L / 2 + scale / 2, L / 2 - scale / 2, N)
    ys = np.linspace(-L / 2 + scale / 2, L / 2 - scale / 2, N)
    x_local, y_local = np.meshgrid(xs, ys)

    # Flatten the coordinate grids for vectorized computation
    x_local_flat = x_local.flatten()
    y_local_flat = y_local.flatten()

    # Expand local coordinates to batch size
    x_local_flat_expanded = np.tile(x_local_flat, (K, 1))  # Shape (K, N*N)
    y_local_flat_expanded = np.tile(y_local_flat, (K, 1))  # Shape (K, N*N)

    # Expand robot positions and orientations to match local coordinates
    x_expanded = x[:, np.newaxis]  # Shape (K, 1)
    y_expanded = y[:, np.newaxis]  # Shape (K, 1)
    cos_theta = np.cos(theta)[:, np.newaxis]  # Shape (K, 1)
    sin_theta = np.sin(theta)[:, np.newaxis]  # Shape (K, 1)

    # Compute global coordinates for all robots in the batch (vectorized)
    x_global_flat = cos_theta * x_local_flat_expanded - sin_theta * y_local_flat_expanded + x_expanded  # Shape (K, N*N)
    y_global_flat = sin_theta * x_local_flat_expanded + cos_theta * y_local_flat_expanded + y_expanded  # Shape (K, N*N)

    # using original env method to get indices
    y_indices = np.floor((map_center[1] - y_global_flat) / s_global).astype(int)
    x_indices = np.floor((x_global_flat + map_center[0]) / s_global).astype(int)

    # Handle boundary conditions to prevent index out of bounds
    x_indices = np.clip(x_indices, 0, global_map.shape[1] - 1)  # Assuming shape of global_maps is (K, H, W)
    y_indices = np.clip(y_indices, 0, global_map.shape[0] - 1)

    # Extract occupancy values from the global maps for each robot (vectorized)
    occupancy_values = global_map[y_indices, x_indices]  # Shape (K, N*N)

    # Reshape the occupancy values to form the local maps for each robot
    local_maps = occupancy_values.reshape(K, N, N)  # Shape (K, N, N)

    return local_maps


def png_to_binary_csv(image_path, csv_path, downsample_factor=1):
    """
    Converts a PNG image to a binary CSV file.
    White pixels (RGB 255,255,255) are converted to 0, all others to 1.
    Allows downsampling by a given factor before saving.

    :param image_path: Path to the input PNG image.
    :param csv_path: Path to save the output CSV file.
    :param downsample_factor: Factor by which to downsample the image (default is 1, no downsampling).
    """
    # Open the image and convert it to RGB
    img = Image.open(image_path).convert('RGB')

    # Downsample the image if required
    if downsample_factor > 1:
        new_size = (img.width // downsample_factor, img.height // downsample_factor)
        img = img.resize(new_size)

    img_array = np.array(img)

    # Create a binary matrix: 0 for white pixels, 1 for all others
    binary_matrix = np.where(np.all(img_array == [255, 255, 255], axis=-1), 0, 1)

    # Save to CSV using NumPy
    np.savetxt(csv_path, binary_matrix, fmt='%d', delimiter=',')

    print(f"Binary CSV saved to {csv_path}")

if __name__ == "__main__":
    # Test the is_colliding function
    import gymnasium as gym
    import matplotlib.pyplot as plt
    import torch
    # from local_map_encoder import GridEncoder, MaxEncoder

    png_to_binary_csv("../maps/mazes/boxes.png.", "../maps/mazes/boxes.csv")
    png_to_binary_csv("../maps/mazes/shapes.png", "../maps/mazes/shapes.csv")
    png_to_binary_csv("../maps/mazes/narrow.png.", "../maps/mazes/narrow.csv")
    png_to_binary_csv("../maps/mazes/narrow_short.png.", "../maps/mazes/narrow_short.csv")

    convert_forests_to_maps('../maps/forests')

    apartment_map = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Top wall
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Room boundary
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],  # Wall division for rooms
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],  # Room and hallway
        [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],  # Open space
        [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1],  # Wall boundaries
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # Hallway space
        [1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # More wall divisions
        [1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],  # Rooms with openings
        [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],  # Similar to above
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Open floor
        [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1],  # Wall divisions
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # More open floor
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Bottom wall
    ]

    # Create a maze environment
    env_id = 'PointMaze_Large-v3'
    env = gym.make(env_id, render_mode='human', max_episode_steps=1000)
    env.reset()

    global_map = np.float32(env.maze.maze_map)
    map_center = env.maze.x_map_center, env.maze.y_map_center
    s_global = env.maze.maze_size_scaling

    # Define the state of the ball
    # state = np.float32([-1.6519535, 0.54289424, 0.0, 0.0])
    state = np.float32([-3.461082, 1.40738, 0.0, 0.0])
    env.point_env.set_state(state[:2], state[2:])
    env.render()

    # Plot the maze
    plt.imshow(global_map, cmap="binary", origin='lower',
               extent=[-map_center[0], map_center[0], map_center[1], -map_center[1]])
    plt.scatter(state[0], state[1], color="green", s=100)
    plt.show()

    # Test the create_local_map function
    x, y, theta = state[0], state[1], 0.0
    map_size = (10, 10)
    scale = 0.2
    local_map = create_local_map(global_map, x, y, theta, map_size, scale, s_global, map_center)
    # Plot the local map
    plt.imshow(local_map[0], cmap="binary")
    plt.scatter(map_size[0] // 2, map_size[1] // 2, color="green", s=100)
    plt.show()

    # Check for collisions
    collision = is_colliding_maze(env, state)
    print("Collision:", collision)

    # Define the state of the ball
    state = np.float32([0.5, 0.5, 0.0, 0.0])
    env.point_env.set_state(state[:2], state[2:])
    env.render()

    # Plot the maze
    plt.imshow(global_map, cmap="binary", origin='lower',
               extent=[-map_center[0], map_center[0], map_center[1], -map_center[1]])
    plt.scatter(state[0], state[1], color="green", s=100)
    plt.show()

    # Test the create_local_map function
    x, y, theta = state[0], state[1], 0.0
    map_size = (10, 10)
    scale = 0.2
    local_map = create_local_map(global_map, x, y, theta, map_size, scale, s_global, map_center)
    # Plot the local map
    plt.imshow(local_map[0], cmap="binary")
    plt.scatter(map_size[0] // 2, map_size[1] // 2, color="green", s=100)
    plt.show()

    # Test the create_local_map function
    x, y, theta = 0.5, 0.5, np.pi / 4
    map_size = (20, 20)
    scale = 0.1
    local_map = create_local_map(global_map, x, y, theta, map_size, scale, s_global, map_center)
    # Plot the local map
    plt.imshow(local_map, cmap="binary")
    plt.scatter(map_size[0] // 2, map_size[1] // 2, color="green", s=100)
    plt.show()

    # Test the create_local_map function
    x, y, theta = 0.5, 0.5, np.pi / 4
    map_size = (10, 10)
    scale = 0.2
    local_map = create_local_map(global_map, x, y, theta, map_size, scale, s_global, map_center)
    # Plot the local map
    plt.imshow(local_map, cmap="binary")
    plt.scatter(map_size[0] // 2, map_size[1] // 2, color="green", s=100)
    plt.show()

    map_encoder_net = MaxEncoder(
        embedding_dim=9
    )
    encoded_map = map_encoder_net(torch.tensor(local_map).unsqueeze(0))
    plt.imshow(encoded_map.squeeze(0).detach().numpy().reshape(3, 3))
    plt.show()

    # Test the create_local_map function
    x, y, theta = 0.5, 0.5, np.pi / 4
    map_size = (7, 7)
    scale = 0.3
    local_map = create_local_map(global_map, x, y, theta, map_size, scale, s_global, map_center)
    # Plot the local map
    plt.imshow(local_map, cmap="binary")
    plt.scatter(map_size[0] // 2, map_size[1] // 2, color="green", s=100)
    plt.show()
