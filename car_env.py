import math
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.use('TkAgg')  # 'Qt5Agg/'TkAgg'

import gymnasium as gym
from gymnasium import spaces

from casadi import *

from common.map_utils import is_colliding_car


class CarEnv(gym.Env):
    """
    Custom Gym environment for a car.
    """

    def __init__(self, dt=0.02, drone_radius=0.1, maze_map=None,collision_checking=True):
        super(CarEnv, self).__init__()

        # Simulation parameters
        self.dt = 1.0 / 50.0  # 50 Hz
        self.current_step = 0
        self.collision_checking = collision_checking

        # Car parameters
        self.ball_radius = drone_radius #*0.1 # changed for real world experiment

        # Initialize the bicycle model
        self.model, self.constraints = bicycle_model()
        self.state_dim = self.model.x.shape[0]  # 6 states
        self.action_dim = self.model.u.shape[0]  # 2 actions

        # Define car parameters
        self.car_length = 0.35  #0.2
        self.car_width = 0.2    #0.07
        self.m = 0.043  # mass [kg]
        self.C1 = 0.5  # cornering stiffness factor
        self.C2 = 15.5  # cornering stiffness factor
        self.Cm1 = 0.28  # drive force gain
        self.Cm2 = 0.05  # drive force speed-proportional factor
        self.Cr0 = 0.011  # rolling resistance constant
        self.Cr2 = 0.006  # quadratic rolling resistance factor

        self.max_speed = 10 # Maximum speed [m/s]

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([
                self.model.dthrottle_min,
                self.model.ddelta_min
            ]),
            high=np.array([
                self.model.dthrottle_max,
                self.model.ddelta_max
            ]),
            dtype=np.float32
        )
        # [Throttle/brake derivative command, Steering angle derivative command]

        self.observation_space = spaces.Box(
            low=-np.inf,  # Adjust this as needed based on your model's state bounds
            high=np.inf,  # Adjust this as needed based on your model's state bounds
            shape=(self.state_dim,),
            dtype=np.float32
        )
        # [X position, Y position, Heading (yaw), Speed, Throttle/brake command, Steering angle]

        # Initial state of the system
        self.v_xy = True  # use v_xy instead of v and theta
        self.state = self.model.x0

        self._maze_map = maze_map
        self._maze_height = 1
        self._maze_size_scaling = 1
        self._map_length = len(maze_map)
        self._map_width = len(maze_map[0])
        self._x_map_center = self._map_width / 2 * self._maze_size_scaling
        self._y_map_center = self._map_length / 2 * self._maze_size_scaling

        # Initialize state and goal
        self.goal = np.array([0, 0])
        self.done = False
        self.terminated = False

        # Rendering objects
        self.fig = None
        self.ax = None
        plt.ion()  # Enable interactive mode

    @property
    def maze_map(self) -> List[List[Union[str, int]]]:
        """Returns the list[list] data structure of the maze."""
        return self._maze_map

    @property
    def maze_size_scaling(self) -> float:
        """Returns the scaling value used to integrate the maze
        encoding in the MuJoCo simulation.
        """
        return self._maze_size_scaling

    @property
    def maze_height(self) -> float:
        """Returns the un-scaled height of the walls in the MuJoCo
        simulation.
        """
        return self._maze_height

    @property
    def x_map_center(self) -> float:
        """Returns the x coordinate of the center of the maze in the MuJoCo simulation"""
        return self._x_map_center

    @property
    def y_map_center(self) -> float:
        """Returns the x coordinate of the center of the maze in the MuJoCo simulation"""
        return self._y_map_center

    def cell_rowcol_to_xy(self, rowcol_pos: np.ndarray) -> np.ndarray:
        """Converts a cell index `(i,j)` to x and y coordinates in the MuJoCo simulation"""
        x = (rowcol_pos[1] + 0.5) * self.maze_size_scaling - self.x_map_center
        y = self.y_map_center - (rowcol_pos[0] + 0.5) * self.maze_size_scaling

        return np.array([x, y])

    def cell_xy_to_rowcol(self, xy_pos: np.ndarray) -> np.ndarray:
        """Converts a cell x and y coordinates to `(i,j)`"""
        i = math.floor((self.y_map_center - xy_pos[1]) / self.maze_size_scaling)
        j = math.floor((xy_pos[0] + self.x_map_center) / self.maze_size_scaling)
        return np.array([i, j])

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[Dict[str, Optional[np.ndarray]]] = None,
              **kwargs, ):
        """
        Reset the environment to an initial state.
        """
        # Initial state: random position, zero velocity, neutral orientation
        # self.current_pose = np.array([.0, .0, .0, .0, .0, .0])
        self.state = np.zeros(self.model.x.shape[0], dtype=np.float32)
        # self.state[0:2] = 0  # random position
        # self.state[2] = 0.5  # initial height
        # self.state[3:6] = 0  # zero velocity
        # self.state[6:10] = np.array([1.0, 0.0, 0.0, 0.0])  # neutral quaternion
        # Random goal position
        # self.goal = np.random.uniform(-5, 5, size=3)
        if options is not None:
            if "goal_cell" in options and options["goal_cell"] is not None:
                self.goal = self.cell_rowcol_to_xy(options["goal_cell"])

            if "reset_cell" in options and options["reset_cell"] is not None:
                self.state[0:2] = self.cell_rowcol_to_xy(options["reset_cell"])
            if "reset_deg" in options and options["reset_deg"] is not None:
                self.state[2] = np.deg2rad(options["reset_deg"])
        self.current_step = 0
        self.done = False
        self.terminated = False

        # initial_noise = np.random.uniform(-0.2, 0.2, size=6)
        # self.state[0:6] += initial_noise

        return self._get_obs(), None

    def step(self, action):
        """
        Take one step in the environment using the given action.

        Parameters:
        - action: np.ndarray of shape (4,), [F, w_x, w_y, w_z]

        Returns:
        - observation: np.ndarray, the next state of the drone
        - reward: float, the reward for this step
        - done: bool, whether the episode is over
        - info: dict, additional information
        """
        collision = False
        if not self.done and not self.terminated:

            # Update state using drone dynamics
            self.state = self._update_state(self.state, action)

            # Calculate reward
            reward = self._calculate_reward()

            # Check if the episode is done
            self.current_step += 1
            self.done = self._check_done()

            # check if terminated (due to collision)
            if self.collision_checking:
                collision = is_colliding_car(self.state[:3], self.maze_map)

            if collision:
                reward = -1.0
                self.terminated = True

        else:
            reward = 0.0

        # Return observation, reward, done, info
        # obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx]
        # return self._get_obs(), reward, self.done, False, collision
        # info = {"collision": collision, "goal": self.goal, "done": self.done}
        info = {"collision": collision, "goal": self.goal, "success": self.done}
        return self._get_obs(), reward, self.terminated, False, info

    def render(self, mode='human'):
        """
        Render the environment. Currently, this is a placeholder.
        """
        if mode == 'human':
            if self.fig is None or self.ax is None:
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(111, projection='3d')

            self.ax.clear()
            self.ax.set_xlabel('X [m]')
            self.ax.set_ylabel('Y [m]')
            self.ax.set_zlabel('Z [m]')
            self.ax.set_title("Drone Trajectory with Maze")
            self.ax.set_xlim([-self.x_map_center, self.x_map_center])
            self.ax.set_ylim([-self.y_map_center, self.y_map_center])
            self.ax.set_zlim([0, 2])

            self.draw_maze()
            self._draw_car()
            plt.pause(0.001)

    def set_state(self, state):
        # if self.v_xy:
        #     v_x = state[2]
        #     v_y = state[3]
        #     state[2] = np.arctan2(v_y, v_x)
        #     state[3] = np.sqrt(v_x ** 2 + v_y ** 2)
        self.state = state

    def _get_obs(self):
        """
        Return the current observation (state).
        """
        # obs = {
        #     'observation': self.state.copy(),
        #     'desired_goal': self.goal.copy(),
        # }
        state = self.state.copy()
        # if self.v_xy:
        #     theta = state[2]
        #     v = state[3]
        #     state[2] = v * np.cos(theta)
        #     state[3] = v * np.sin(theta)
        return state

    def _calculate_reward(self):
        """
        Calculate the reward based on the current state and goal.
        Reward is higher when closer to the goal.
        """
        # position = self.state[0:2]
        # distance_to_goal = np.linalg.norm(position - self.goal)

        # Negative reward proportional to the distance to the goal
        return 0

    def _check_done(self):
        """
        Check whether the episode is done.
        """
        position = self.state[0:2]
        distance_to_goal = np.linalg.norm(position - self.goal)

        # End the episode if the drone is close enough to the goal or max steps exceeded
        if distance_to_goal < 0.5:
            return True
        # if self.current_step >= self.max_steps:
        #     return True

        return False

    def _update_state(self, state, action):
        """
        Update the state of the car based on the control inputs using dynamics.

        Parameters:
        - state: np.ndarray, the current state of the car
        - F: float, the thrust force
        - w: np.ndarray of shape (3,), the angular rates

        Returns:
        - new_state: np.ndarray, the updated state
        """
        state = self.state

        # Ensure the action is within the action space bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Compute state derivatives using the model's explicit dynamics
        # f_expl = self.model.f_expl_expr
        # state_dot = f_expl(state, action).full().flatten()
        psi = state[2]
        v = state[3]
        D = state[4]
        steer_angle = state[5]
        Fxd = (self.Cm1 - self.Cm2 * v) * D - self.Cr2 * (v ** 2) - self.Cr0 * MX.tanh(5.0 * v)
        state_dot = np.array([
            v * np.cos(psi + self.C1 * steer_angle),  # dx/dt
            v * np.sin(psi + self.C1 * steer_angle),  # dy/dt
            v * self.C2 * steer_angle,  # dpsi/dt
            (Fxd / self.m) * np.cos(self.C1 * steer_angle),  # dv/dt
            action[0],  # dD/dt
            action[1]  # d(delta)/dt
        ])
        # Integrate state using Euler's method
        next_state = state + self.dt * state_dot

        next_state[3] = np.clip(next_state[3], 0, self.max_speed)  # Limit speed to max_speed
        # Update state
        self.state = next_state

        # Return step information
        return next_state

    def draw_maze(self):

        for i in range(self._map_length):
            for j in range(self._map_width):
                if self.maze_map[i][j] == 1:
                    x, y = self.cell_rowcol_to_xy([i, j])
                    x -= 0.5
                    y -= 0.5
                    self.ax.bar3d(x, y, 0, 1, 1, 1, color='grey', alpha=0.8, shade=True)

    def _draw_car(self):

        # Extract car position and orientation
        x, y, psi = self.state[0], self.state[1], self.state[2]

        # Calculate box corners
        car_corners = np.array([
            [self.car_length / 2, self.car_width / 2],
            [self.car_length / 2, -self.car_width / 2],
            [-self.car_length / 2, -self.car_width / 2],
            [-self.car_length / 2, self.car_width / 2]
        ])

        # Rotate and translate the box
        rotation_matrix = np.array([
            [np.cos(psi), -np.sin(psi)],
            [np.sin(psi), np.cos(psi)]
        ])
        transformed_corners = np.dot(car_corners, rotation_matrix.T) + np.array([x, y])

        # Draw the car as a box
        self.ax.plot(
            [transformed_corners[0, 0], transformed_corners[1, 0],
             transformed_corners[2, 0], transformed_corners[3, 0],
             transformed_corners[0, 0]],
            [transformed_corners[0, 1], transformed_corners[1, 1],
             transformed_corners[2, 1], transformed_corners[3, 1],
             transformed_corners[0, 1]],
            "b-", linewidth=2
        )

        # Draw direction vector
        direction_x = np.cos(psi) * self.car_length / 2
        direction_y = np.sin(psi) * self.car_length / 2
        self.ax.quiver(
            x, y, 0, direction_x, direction_y, 0,
             color='red'
        )

        # self.ax.bar3d(
        #     car_x, car_y, car_z,  # Bottom corner of the bar
        #     self.car_length, self.car_width, 0.1,  # Dimensions
        #     color='blue', alpha=0.7
        # )
        # # Draw the direction vector as a quiver
        # direction_x = np.cos(psi)
        # direction_y = np.sin(psi)
        # self.ax.quiver(
        #     x, y, 0.05,  # Starting point of the vector
        #     direction_x, direction_y, 0,  # Direction of the vector
        #     length=1.0, color='red', arrow_length_ratio=0.2
        # )



#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

# author: Daniel Kloeser
def bicycle_model():
    """
    Adapted bicycle model in a standard 2D Cartesian state space:
      States:  x, y, psi, v, D, delta
        x     = global X position [m]
        y     = global Y position [m]
        psi   = heading (yaw) [rad]
        v     = forward speed [m/s]
        D     = throttle (or brake) command
        delta = steering angle [rad] (positive is left)

      Controls: dD, dDelta
        dD     = time derivative of the throttle/brake [-2, 2]
        dDelta = time derivative of the steering angle [-10, 10] [rad/s]


    """
    # We'll store relevant information in these two objects:
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()

    model_name = "CartesianBicycleModel"

    # --- Vehicle / tire parameters ---
    m = 0.043  # mass [kg]
    C1 = 0.5  # cornering stiffness factor
    C2 = 15.5  # cornering stiffness factor
    Cm1 = 0.28  # drive force gain
    Cm2 = 0.05  # Velocity-dependent damping [kg/s]
    Cr0 = 0.011  # Static rolling resistance coefficient [kg·m/s²]
    Cr2 = 0.006  # quadratic drag coefficient [kg/m]

    # --- CasADi symbols for states ---
    x_sym = MX.sym("x")  # X position
    y_sym = MX.sym("y")  # Y position
    psi_sym = MX.sym("psi")  # Heading (yaw)
    v_sym = MX.sym("v")  # Speed
    D_sym = MX.sym("D")  # Throttle/brake command
    delta_sym = MX.sym("delta")  # Steering angle
    x = vertcat(x_sym, y_sym, psi_sym, v_sym, D_sym, delta_sym)

    # --- CasADi symbols for controls ---
    dD_sym = MX.sym("dD")  # time derivative of throttle
    dDelta_sym = MX.sym("dDelta")  # time derivative of steering
    u = vertcat(dD_sym, dDelta_sym)

    # --- Time derivatives for states (xdot) ---
    x_dot_sym = MX.sym("x_dot")
    y_dot_sym = MX.sym("y_dot")
    psi_dot_sym = MX.sym("psi_dot")
    v_dot_sym = MX.sym("v_dot")
    D_dot_sym = MX.sym("D_dot")
    delta_dot_sym = MX.sym("delta_dot")
    xdot = vertcat(x_dot_sym, y_dot_sym, psi_dot_sym, v_dot_sym, D_dot_sym, delta_dot_sym)

    # algebraic variables
    z = vertcat([])

    # parameters
    p = vertcat([])

    # --- Longitudinal force ---
    #   Fxd = (Cm1 - Cm2 * v) * D - Cr2 * v^2 - Cr0 * tanh(5*v)
    Fxd = (Cm1 - Cm2 * v_sym) * D_sym - Cr2 * (v_sym ** 2) - Cr0 * MX.tanh(5.0 * v_sym)
    f_expl = vertcat(
        v_sym * MX.cos(psi_sym + C1 * delta_sym),  # dx/dt
        v_sym * MX.sin(psi_sym + C1 * delta_sym),  # dy/dt
        v_sym * C2 * delta_sym,  # dpsi/dt
        (Fxd / m) * MX.cos(C1 * delta_sym),  # dv/dt
        dD_sym,  # dD/dt
        dDelta_sym  # d(delta)/dt
    )

    # --- Lateral and longitudinal accelerations (for constraints) ---
    # a_long = Fxd / m
    # a_lat  = v^2 * C2 * delta + (Fxd/m)*sin(C1*delta)
    a_long = Fxd / m
    a_lat = (v_sym ** 2) * C2 * delta_sym + (Fxd / m) * MX.sin(C1 * delta_sym)

    # --- Collect model / constraints info ---
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = z
    model.p = p
    model.name = model_name

    # Model bounds
    model.n_min = -0.12  # width of the track [m]
    model.n_max = 0.12  # width of the track [m]

    # state bounds
    model.throttle_min = -1.0
    model.throttle_max = 1.0

    model.delta_min = -0.40  # minimum steering angle [rad]
    model.delta_max = 0.40  # maximum steering angle [rad]

    # input bounds
    model.ddelta_min = -2.0  # minimum change rate of stering angle [rad/s]
    model.ddelta_max = 2.0  # maximum change rate of steering angle [rad/s]
    model.dthrottle_min = -10  # -10.0  # minimum throttle change rate
    model.dthrottle_max = 10  # 10.0  # maximum throttle change rate

    # nonlinear constraint
    constraint.alat_min = -4  # maximum lateral force [m/s^2]
    constraint.alat_max = 4  # maximum lateral force [m/s^1]

    constraint.along_min = -4  # maximum lateral force [m/s^2]
    constraint.along_max = 4  # maximum lateral force [m/s^2]

    # Define initial conditions
    model.x0 = np.array([0, 0, 0, 0, 0, 0])

    # define constraints struct
    constraint.alat = Function("a_lat", [x, u], [a_lat])
    constraint.expr = vertcat(a_long, a_lat, D_sym, delta_sym)
    constraint.alat = Function("a_lat", [x, u], [a_lat])

    # Save parameters in a small container
    params = types.SimpleNamespace()
    params.m = m
    params.C1 = C1
    params.C2 = C2
    params.Cm1 = Cm1
    params.Cm2 = Cm2
    params.Cr0 = Cr0
    params.Cr2 = Cr2
    model.params = params
    model.name = model_name
    return model, constraint
