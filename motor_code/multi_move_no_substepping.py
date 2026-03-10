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

# forward kinematics
def forward_kinematics(lengths):
    x0, y0 = corner_positions[0]
    l0 = lengths[0]
    A, b = [], []
    for i in [1, 2, 3]:
        xi, yi = corner_positions[i]
        li = lengths[i]
        A.append([2*(xi - x0), 2*(yi - y0)])
        b.append((l0**2 - li**2) + (xi**2 - x0**2) + (yi**2 - y0**2))
    res = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    return res[0]

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

def evaluate_spline_norm(coeffs, tau, dt):
    """Evaluates position, real-world velocity, and real-world acceleration."""
    c0, c1, c2, c3, c4, c5 = coeffs
    
    # Position 
    p = c0 + c1*tau + c2*tau**2 + c3*tau**3 + c4*tau**4 + c5*tau**5
    
    # Velocity: (1/dt) * dp/dtau
    v_tau = c1 + 2*c2*tau + 3*c3*tau**2 + 4*c4*tau**3 + 5*c5*tau**4
    v = v_tau / dt
    
    # Acceleration: (1/dt^2) * d2p/dtau2
    a_tau = 2*c2 + 6*c3*tau + 12*c4*tau**2 + 20*c5*tau**3
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
        next_p, _, _ = evaluate_spline_norm(coeffs, 1.0, dt)

        # get analytical lengths at this timestep
        next_lengths = np.linalg.norm(corner_positions - next_p, axis=1)
        
        # dl is the length difference vector from the timestep
        dl = next_lengths - curr_lengths
        
        # store data
        pos_history[i + 1] = next_p
        motor_history[i] = dl
        
        # handoff for next iteration
        curr_p, curr_v, curr_a = evaluate_spline_norm(coeffs, 1.0, dt)
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

def compute_path_from_motor(start, motor_history):
    # start pos is x, y vector
    curr_pos = start.copy()
    curr_lengths = np.linalg.norm(corner_positions - curr_pos, axis=1)

    # write data
    pos_history = np.zeros((len(motor_history) + 1, 2))
    pos_history[0] = curr_pos

    for i in range(len(motor_history)):
        curr_lengths += motor_history[i]
        curr_position = forward_kinematics(curr_lengths)
        pos_history[i+1] = curr_position

    return pos_history

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
                        init_func=init, blit=True, interval=time/ticks) #time/ticks)

    plt.legend()
    plt.show()

def animate_trajectory_compare(pos_target, pos_actual, motor_history, time, smoothing=False, labels=("Target", "Actual")):
    # 1. Calculate Deviation
    # We take the Euclidean distance between target and actual at every tick
    # Ensure they are the same length
    min_len = min(len(pos_target), len(pos_actual))
    deviations = np.linalg.norm(pos_target[:min_len] - pos_actual[:min_len], axis=1)
    max_dev = np.max(deviations)
    avg_dev = np.mean(deviations)
    
    ticks = pos_actual.shape[0]
    ticks_per_sec = int(ticks / time)
    
    # Use analytical velocity for the plot if possible, else numerical diff
    cartesian_vels = np.linalg.norm(np.diff(pos_actual, axis=0), axis=1) * ticks_per_sec
    
    fig, (ax_rob, ax_mot, ax_vel) = plt.subplots(1, 3, figsize=(18, 6))

    # --- LEFT PLOT: Workspace ---
    ax_rob.set_xlim(-50, 550); ax_rob.set_ylim(-50, 550)
    ax_rob.set_aspect('equal')
    ax_rob.set_title(f"Max Deviation: {max_dev:.4f} mm")
    
    ax_rob.scatter(corner_positions[:,0], corner_positions[:,1], c='red', marker='s', s=80)
    
    # Target Path (Ghost)
    ax_rob.plot(pos_target[:,0], pos_target[:,1], 'k--', alpha=0.2, label=labels[0])
    
    effector, = ax_rob.plot([], [], 'bo', markersize=8, label=labels[1])
    trace, = ax_rob.plot([], [], 'b-', linewidth=1, alpha=0.6)
    cables = [ax_rob.plot([], [], 'k-', linewidth=1, alpha=0.3)[0] for _ in range(4)]
    ax_rob.legend(loc='upper right', fontsize='x-small')

    # --- MIDDLE PLOT: Motors ---
    ax_mot.set_title("Motor Step Increments (dl)")
    mot_lines = [ax_mot.plot([], [], label=f"M{i}")[0] for i in range(4)]
    ax_mot.set_xlim(0, len(motor_history))
    d_limit = np.max(np.abs(motor_history)) * 1.5
    ax_mot.set_ylim(-d_limit, d_limit)
    ax_mot.legend(loc='upper right', fontsize='x-small')

    # --- RIGHT PLOT: Velocity & Deviation ---
    ax_vel.set_title("Velocity & Error")
    vel_line, = ax_vel.plot([], [], 'g-', label="Vel (mm/s)")
    err_line, = ax_vel.plot([], [], 'r:', alpha=0.6, label="Dev (mm)")
    ax_vel.set_xlim(0, len(pos_actual))
    ax_vel.set_ylim(0, max(np.max(cartesian_vels), max_dev) * 1.1)
    ax_vel.legend(loc='upper right', fontsize='x-small')

    def update(frame):
        # Update Robot
        curr_p = pos_actual[frame]
        effector.set_data([curr_p[0]], [curr_p[1]])
        trace.set_data(pos_actual[:frame, 0], pos_actual[:frame, 1])
        for i, cable in enumerate(cables):
            px, py = corner_positions[i]
            cable.set_data([px, curr_p[0]], [py, curr_p[1]])
            
        # Update Motors & Velocity/Error
        if frame > 0:
            t = np.arange(frame)
            for i, line in enumerate(mot_lines):
                line.set_data(t, motor_history[:frame, i])
            vel_line.set_data(t, cartesian_vels[:frame])
            err_line.set_data(t, deviations[:frame])
            
        return [effector, trace, vel_line, err_line] + cables + mot_lines

    ani = FuncAnimation(fig, update, frames=min_len, blit=True, interval=time/ticks)
    plt.tight_layout()
    plt.show()
    
    print(f"--- Analysis Results ---")
    print(f"Max Deviation: {max_dev:.6f} mm")
    print(f"Mean Error:    {avg_dev:.6f} mm")

def draw_line(start_pos, end_pos, start_v_vec, end_v_vec, duration, dt=1/20):
    dist = np.linalg.norm(end_pos - start_pos)
    if dist < 1e-6: return np.array([[start_pos[0], start_pos[1], 0, 0]])
    
    unit_dir = (end_pos - start_pos) / dist


    # project 2D velocity vectors onto the 1D line path
    v0_s = np.dot(start_v_vec, unit_dir)
    v1_s = np.dot(end_v_vec, unit_dir)
    
    # reuse your existing solver for 1D scalar 's'
    # assume start/end acceleration is 0 for the segment
    # arguments here are scalars because we are doing a line segment
    coeffs_s = get_quintic_coeffs_norm(0, v0_s, 0, dist, v1_s, 0, duration)
    
    num_steps = int(duration / dt)
    t_ticks = np.linspace(0, duration, num_steps)
    
    # vectorized evaluation using your existing evaluate_spline logic
    # assuming evaluate_spline returns p, v, a
    s_path = np.array([evaluate_spline_norm(coeffs_s, t/duration, duration) for t in t_ticks])
    s = s_path[:, 0]
    ds_dt = s_path[:, 1]
    
    # 4. Map back to 2D Cartesian
    px = start_pos[0] + s * unit_dir[0]
    py = start_pos[1] + s * unit_dir[1]
    vx = ds_dt * unit_dir[0]
    vy = ds_dt * unit_dir[1]
    
    return np.column_stack((px, py, vx, vy))

def draw_arc(center, start_pos, radians, start_v_vec, end_v_vec, duration, dt=1/20):
    rel_start = start_pos - center
    radius = np.linalg.norm(rel_start)
    start_theta = np.arctan2(rel_start[1], rel_start[0])
    
    # 1. Start Angular Velocity: 
    # Tangent is perpendicular to the radius vector
    tangent0 = np.array([-rel_start[1], rel_start[0]]) / radius
    omega0 = np.dot(start_v_vec, tangent0) / radius
    
    # 2. End Angular Velocity:
    # We rotate the start radius by the arc 'radians' to find the end tangent
    end_theta = start_theta + radians
    tangent1 = np.array([-np.sin(end_theta), np.cos(end_theta)])
    omega1 = np.dot(end_v_vec, tangent1) / radius
    
    # 3. 1D Scalar Spline for Theta
    # Reuse your scalar solver: p0=0, p1=radians
    coeffs_theta = get_quintic_coeffs_norm(0, omega0, 0, radians, omega1, 0, duration)
    
    num_steps = int(duration / dt)
    t_ticks = np.linspace(0, duration, num_steps)
    
    # 4. Evaluate and Map
    # Ensure evaluate_spline_norm takes (coeffs, tau, dt_scaling)
    res = np.array([evaluate_spline_norm(coeffs_theta, t/duration, duration) for t in t_ticks])
    theta_rel = res[:, 0]
    dtheta_dt = res[:, 1]
    
    current_theta = start_theta + theta_rel
    px = center[0] + radius * np.cos(current_theta)
    py = center[1] + radius * np.sin(current_theta)
    
    # Velocity components (Chain Rule)
    vx = -radius * np.sin(current_theta) * dtheta_dt
    vy =  radius * np.cos(current_theta) * dtheta_dt
    
    return np.column_stack((px, py, vx, vy))


# run the thing
sim_duration = 5
sim_steps_per_sec = 100
dt = 1 / sim_steps_per_sec

# circle_data = generate_smooth_circle(center=(250, 250), 
#                 radius=175, duration=sim_duration, dt = 1/sim_steps_per_sec)
# pos_test, motor_test = move_interp_accel_final(circle_data, 120)

# center = np.array([150, 150])

# start_pos = np.array([100, 150])
# end_pos = np.array([450, 400])
# start_v = np.array([0, 0])
# end_v = np.array([0, 0])

# line_data = draw_line(start_pos, end_pos, start_v, end_v, sim_duration, dt)
# pos_test, motor_test = move_interp_accel_final(line_data, sim_steps_per_sec)
# kinematics = compute_path_from_motor(pos_test[0], motor_test)

# arc_data = draw_arc(center, start_pos, np.pi, start_v, end_v, sim_duration, dt)
# pos_test, motor_test = move_interp_accel_final(arc_data, sim_steps_per_sec)



np.set_printoptions(threshold=np.inf, precision=6, suppress=True) # isnt really necessary
# animate_trajectory(pos_test, motor_test, sim_duration, smoothing=False)

def manual_animate(pos_target, pos_actual, time):
    # FORCE BACKEND BEFORE PLOTTING
    plt.switch_backend('TkAgg') # Or 'Qt5Agg' if you have it
    
    plt.ion() # Interactive mode ON
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Pre-plot the ghost path
    ax.plot(pos_target[:,0], pos_target[:,1], 'k--', alpha=0.3, label="Target")
    effector, = ax.plot([], [], 'bo', label="Actual")
    trace, = ax.plot([], [], 'b-', alpha=0.6)
    
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.legend()

    for i in range(len(pos_actual)):
        effector.set_data([pos_actual[i,0]], [pos_actual[i,1]])
        trace.set_data(pos_actual[:i, 0], pos_actual[:i, 1])
        
        plt.draw()
        plt.pause(0.001) # This forces the window to refresh and stay open
        
        # If user closes window, stop loop
        if not plt.fignum_exists(fig.number):
            break

    plt.ioff() # Interactive mode OFF
    plt.show(block=True) # Final block to keep it open

# draw a shape
def draw():
    arc1_center = np.array([125, 100])
    arc1_start = np.array([200, 100])
    zero = np.array([0.0, 0.0])
    velocity = np.array([0.0, 500])
    arc1_data = draw_arc(arc1_center, arc1_start, 2*np.pi, zero, velocity, 0.8, dt)

    line1_end = arc1_start + np.array([0.0, 300]) 
    line1_data = draw_line(arc1_start, line1_end, velocity, velocity, 0.5, dt)

    arc2_center = np.array([250, 400])
    arc2_data = draw_arc(arc2_center, line1_end, -np.pi, velocity, -velocity, 0.4, dt)

    line2_start = line1_end + np.array([100.0, 0.0])
    line2_end = arc1_start + np.array([100, 0.0])
    line2_data = draw_line(line2_start, line2_end, -velocity, -velocity, 0.5, dt)

    arc3_center = np.array([375, 100])
    arc3_data = draw_arc(arc3_center, line2_end, 2*np.pi, -velocity, zero, 0.8, dt)

    total_data = np.vstack([arc1_data, line1_data[1:], arc2_data[1:], line2_data[1:], arc3_data[1:]])

    pos_test, motor_test = move_interp_accel_final(total_data, sim_steps_per_sec)        
    kinematics = compute_path_from_motor(pos_test[0], motor_test)

    animate_trajectory_compare(pos_test, kinematics, motor_test, sim_duration, smoothing=False)

draw()