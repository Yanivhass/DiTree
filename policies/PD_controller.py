import numpy as np


def car_pd_controller(state, target, v_des=10,
                  Kp_steer=2.0, Kd_steer=0.3,
                  Kp_vel=1.0, Kd_vel=0.3,
                  dt=0.01,
                  prev_heading_error=0.0,
                  prev_vel_error=0.0):
    """
    x, y, psi: current pose
    x_ref, y_ref: desired waypoint
    v: current speed
    v_des: desired speed
    Kp_steer, Kd_steer: PD gains for heading
    Kp_vel,   Kd_vel:   PD gains for velocity
    dt: time step
    prev_heading_error, prev_vel_error: needed for 'D' terms

    returns: (delta, D, heading_error, vel_error)
      plus we return the updated heading & velocity errors so that in your loop
      you can store them for the next iteration.
    """
    if len(state.shape) == 1:
        state = np.expand_dims(state, axis=0)
    if len(target.shape) == 1:
        target = np.expand_dims(target, axis=0)
    x = state[:,0]
    y = state[:,1]
    psi = state[:,2]
    v = state[:,3]
    x_ref = target[:,0]
    y_ref = target[:,1]
    # 1) Heading to waypoint
    heading_des = np.arctan2(y_ref - y, x_ref - x)
    # 2) Heading error
    heading_error = heading_des - psi
    # normalize heading_error to (-pi, pi)
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

    # 3) PD law for steering (delta)
    heading_error_dot = (heading_error - prev_heading_error) / dt
    delta_cmd = (Kp_steer * heading_error) + (Kd_steer * heading_error_dot)

    # saturate delta
    delta_cmd = np.clip(delta_cmd, -2, 2)

    # 4) Speed error
    vel_error = v_des - v
    vel_error_dot = (vel_error - prev_vel_error) / dt
    D_cmd = Kp_vel * vel_error + Kd_vel * vel_error_dot

    # saturate D in [-10, 10]
    D_cmd = np.clip(D_cmd, -10, 10)

    return D_cmd,delta_cmd, heading_error, vel_error