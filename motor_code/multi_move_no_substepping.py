import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# config
width, height = 500, 500
corner_positions = np.array([
    [0, 0],       # bottom left
    [0, width],    # bottom right
    [height, 0],   # top left
    [height, width] # top right
])

# trajectory computation (splines)
def compute_a1(p0, p1, p2, p3, v0, v1, v2, v3, a0, dt):
    """Calculates the target acceleration at the next waypoint for C2 continuity."""
    dt2 = dt**2
    # Tridiagonal system components for quintic splines
    d1 = (20/dt2)*(p2 - 2*p1 + p0) - (2/dt)*(3*v2 + 4*v1 + 3*v0) + 3*a0
    d2 = (20/dt2)*(p3 - 2*p2 + p1) - (2/dt)*(3*v3 + 4*v2 + 3*v1) 
    a1_raw = (4 * d1 - d2) / 15
    return np.clip(a1_raw, -10000, 10000) # Safety cap for motor limits

def get_quintic_coeffs_norm(p0, v0, a0, p1, v1, a1, dt):
    """Returns coefficients for p(tau) = c0 + c1*tau + ... where tau is [0, 1]."""
    # Transform real-world units to tau-units (non-dimensionalize)
    v0_n, v1_n = v0 * dt, v1 * dt
    a0_n, a1_n = a0 * (dt**2), a1 * (dt**2)

    c0 = p0
    c1 = v0_n
    c2 = a0_n / 2.0
    
    # Boundary condition gaps
    b_p = p1 - (c0 + c1 + c2)
    b_v = v1_n - (c1 + 2*c2)
    b_a = a1_n - (2*c2)

    # Matrix inverse for tau in [0, 1]
    c3 = 10*b_p - 4*b_v + 0.5*b_a
    c4 = -15*b_p + 7*b_v - b_a
    c5 = 6*b_p - 3*b_v + 0.5*b_a

    return [c0, c1, c2, c3, c4, c5]

def evaluate_spline_norm(coeffs, dt):
    """Evaluates position, real-world velocity, and real-world acceleration."""
    c0, c1, c2, c3, c4, c5 = coeffs
    
    # Position (independent of dt scaling, evaluated at 1)
    p = c0 + c1 + c2 + c3 + c4 + c5
    
    # Velocity: (1/dt) * dp/dtau
    v_tau = c1 + 2*c2 + 3*c3 + 4*c4 + 5*c5
    v = v_tau / dt
    
    # Acceleration: (1/dt^2) * d2p/dtau2
    a_tau = 2*c2 + 6*c3 + 12*c4 + 20*c5
    a = a_tau / (dt**2)
    
    return p, v, a

# main path integrator
def move_interp_accel_final(data, segments_per_sec):
    
    # setup
    dt = 1 / segments_per_sec 
    
    p_data, v_data = data[:, 0:2], data[:, 2:4]
    total_sim_steps = (len(p_data))

    # pad position, velocity data for lookahead (assume static end state)
    p_data = np.vstack((p_data, np.array([p_data[-1, :], p_data[-1, :], p_data[-1, :]])))
    v_data = np.vstack((v_data, np.zeros((3, 2))))

    pos_history = np.zeros((total_sim_steps + 1, 2))
    motor_history = np.zeros((total_sim_steps, 4))

    # initialization
    curr_p, curr_v, curr_a = p_data[0], v_data[0], np.array([0.0, 0.0])
    curr_lengths = np.linalg.norm(corner_positions - curr_p, axis=1)
    pos_history[0] = curr_p

    for i in range(total_sim_steps):
        p1, v1 = p_data[i+1], v_data[i+1]
        p2, v2 = p_data[i+2], v_data[i+2]
        p3, v3 = p_data[i+3], v_data[i+3]

        # compute acceleration via 3-step lookahead
        a1 = compute_a1(curr_p, p1, p2, p3, curr_v, v1, v2, v3, curr_a, dt)
        
        # spline generation
        coeffs = get_quintic_coeffs_norm(curr_p, curr_v, curr_a, p1, v1, a1, dt)

        # get analytical position from the spline
        next_p, _, _ = evaluate_spline_norm(coeffs, dt)

        # get analytical lengths at this timestep
        next_lengths = np.linalg.norm(corner_positions - next_p, axis=1)
        
        # dl is the length difference vector from the timestep
        dl = next_lengths - curr_lengths
        
        # store data
        pos_history[i + 1] = next_p
        motor_history[i] = dl
        
        # handoff for next iteration
        curr_p, curr_v, curr_a = evaluate_spline_norm(coeffs, dt)
        curr_lengths = next_lengths
        
    return pos_history, motor_history
    # TODO use motor_history and start position to generate a new trajectory and compare

# trajectory generation
def generate_smooth_circle(center=(250, 250), radius=150, duration=5.0, dt=1/20):
    num_steps = int(duration / dt)
    t = np.linspace(0, 1, num_steps)
    n = 3 # num loops

    # Normalized ramp
    s = 10*t**3 - 15*t**4 + 6*t**5
    theta = s * (2 * np.pi * n)
    
    # Corrected chain rule derivatives
    ds_dt = (30*t**2 - 60*t**3 + 30*t**4) / duration
    dtheta_dt = ds_dt * (2 * np.pi * n)
    
    p_x = center[0] + radius * np.cos(theta)
    p_y = center[1] + radius * np.sin(theta)
    v_x = -radius * np.sin(theta) * dtheta_dt
    v_y =  radius * np.cos(theta) * dtheta_dt
    
    return np.column_stack((p_x, p_y, v_x, v_y))


# auxiliary
def smooth_data(data, window=3):
    return np.convolve(data, np.ones(window)/window, mode='same')

def animate_trajectory(pos_history, motor_history, time, smoothing=False):
    ticks = pos_history.shape[0]
    ticks_per_sec = int(ticks / time)
    dt = time / ticks

    # corrected velocity calculation
    cartesian_vels = np.linalg.norm(np.diff(pos_history, axis=0), axis=1) * ticks_per_sec
    
    # smooth data
    if smoothing:
        for i in range(4):
            motor_history[:, i] = smooth_data(motor_history[:, i])
        cartesian_vels = smooth_data(cartesian_vels)

    fig, (ax_rob, ax_mot, ax_vel) = plt.subplots(1, 3, figsize=(18, 6))

    # left plot: robot motion (directly from traj)
    ax_rob.set_xlim(-50, 550) 
    ax_rob.set_ylim(-50, 550)
    ax_rob.set_aspect('equal')
    ax_rob.grid(True, linestyle=':', alpha=0.6)
    ax_rob.set_title("Robot Workspace")
    
    # draw pulleys
    ax_rob.scatter(corner_positions[:,0], corner_positions[:,1], c='red', marker='s', s=100)
    
    effector, = ax_rob.plot([], [], 'bo', markersize=8)
    cables = [ax_rob.plot([], [], 'k-', linewidth=1, alpha=0.4)[0] for _ in range(4)]
    trace, = ax_rob.plot([], [], 'b--', linewidth=1, alpha=0.3)

    # middle plot: motor length deltas (dl)
    ax_mot.set_xlim(0, len(motor_history))
    
    # center y axis around 0
    d_max = np.max(np.abs(motor_history)) * 1.2 if len(motor_history) > 0 else 1.0
    ax_mot.set_ylim(-d_max, d_max)
    ax_mot.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax_mot.set_ylabel("Delta L (mm / tick)")
    ax_mot.set_xlabel("Tick Number")
    ax_mot.set_title("Motor Step Increments")
    ax_mot.grid(True, alpha=0.3)
    
    mot_line0 = ax_mot.plot([], [], label="Bottom Left")[0]
    mot_line1 = ax_mot.plot([], [], label="Bottom Right")[0]
    mot_line2 = ax_mot.plot([], [], label="Top Left")[0]
    mot_line3 = ax_mot.plot([], [], label="Top Right")[0]

    mot_lines = [mot_line0, mot_line1, mot_line2, mot_line3]

    ax_mot.legend(loc='upper right', fontsize='x-small', ncol=2)

    # right plot: cartesian velocity
    ax_vel.set_xlim(0, len(pos_history))
    v_max = np.max(cartesian_vels) * 1.2 if len(cartesian_vels) > 0 else 1.0
    ax_vel.set_ylim(0, v_max)
    ax_vel.set_title("End-Effector Velocity")
    ax_vel.set_ylabel("Velocity (mm/s)")
    ax_vel.set_xlabel("Tick Number")
    ax_vel.grid(True, alpha=0.3)

    vel_line, = ax_vel.plot([], [], 'g-', linewidth=2, label="Magnitude")
    ax_vel.legend(loc='upper right', fontsize='x-small')

    # animations
    def init():
        effector.set_data([], [])
        trace.set_data([], [])
        for line in cables + mot_lines:
            line.set_data([], [])
        return [effector, trace] + cables + mot_lines

    def update(frame):
        # update robot plot
        curr_x, curr_y = pos_history[frame]
        effector.set_data([curr_x], [curr_y])
        trace.set_data(pos_history[:frame, 0], pos_history[:frame, 1])
        
        for i in range(4):
            px, py = corner_positions[i]
            cables[i].set_data([px, curr_x], [py, curr_y])
            
        # update motor plot
        if frame > 0:
            ticks = np.arange(frame)
            for i in range(4):
                mot_lines[i].set_data(ticks, motor_history[:frame, i])

        # update velocity plot
            vel_line.set_data(np.arange(frame), cartesian_vels[:frame])
        
        return [effector, trace, vel_line] + cables + mot_lines

    ani = FuncAnimation(fig, update, frames=len(pos_history),
                        init_func=init, blit=True, interval= time/ticks)

    plt.legend()
    plt.show()


# run the thing
sim_duration = 5
sim_steps_per_sec = 100

circle_data = generate_smooth_circle(center=(250, 250), radius=175, duration=sim_duration, dt = 1/sim_steps_per_sec)
pos_test, motor_test = move_interp_accel_final(circle_data, 120)

np.set_printoptions(threshold=np.inf, precision=6, suppress=True) # isnt really necessary
animate_trajectory(pos_test, motor_test, sim_duration, smoothing=False)
