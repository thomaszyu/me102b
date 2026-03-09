import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# define corner positions
width, height = 500, 500

bottom_left = np.array([0, 0])
bottom_right = np.array([0, width])
top_left = np.array([height, 0])
top_right = np.array([height, width])

corner_positions = np.array([bottom_left, bottom_right, top_left, top_right])

def forward_kinematics(lengths):
    # solve for position using least squares
    # lengths is given as [L0, L1, L2, L3] where 0 -> bottom left, 1 -> bottom right, etc.

    # use length to bottom left as "reference"
    x0, y0 = corner_positions[0] # these are both 0 so it doesnt really matter but this is for robustness
    l0 = lengths[0]
    
    A = []
    b = []

    for i in [1, 2, 3]:
        xi, yi = corner_positions[i]
        li = lengths[i]

        # we want to solve the system of equations (x - xi)^2 + (y - yi)^2 = li^2 for x, y.
        # we have 4 corners so this is a system of 4 equations for 2 variables.
        # expanding, we have
        # x^2 - 2x xi + xi^2 + y^2 - 2y yi + yi^2 = li^2 (*)
        # but the bottom left pulley has xi = yi = 0 so
        # x^2 - 2x x0 + x0^2 + y^2 - 2y y0 + y0^2 = l0^2 (**)
        # we subtract (*) from (**) and rearrange to get
        # 2x (xi - x0) + (x0^2 - xi^2) + 2y (yi - y0) + (y0^2 - yi^2) = (l0^2 - li^2)
        # rearrange
        # 2x (xi - x0) + 2y (yi - y0) = (l0^2 - li^2) + (xi^2 - x0^2) + (yi^2 - y0^2)
        # so we can write this as a linear equation Ax = b where x = [x; y].
        # this is still overconstrained but we can solve it using least squares

        A.append([2*(xi - x0), 2*(yi - y0)])
        b.append((l0**2 - li**2) + (xi**2 - x0**2) + (yi**2 - y0**2))

    A = np.array(A)
    b = np.array(b)

    # solve least squares for (x, y) vector (called pos)
    res = np.linalg.lstsq(A, b, rcond=None)
    return res[0]

def compute_a1(p0, p1, p2, p3, v0, v1, v2, v3, a0, a3=0):
    # all of the above are 2-vectors
    dt = 1/100 # why not
    dt2 = dt**2

    d1 = (20/dt2) * (p2 - 2*p1 + p0) - (2/dt)*(3*v2 + 4*v1 + 3*v0) + 3*a0
    d2 = (20/dt2) * (p3 - 2*p2 + p1) - (2/dt)*(3*v3 + 4*v2 + 3*v1) + 3*a3

    a1 = (4 * d1 - d2) / 15
    return a1

def get_quintic_coeffs(p0, v0, a0, p1, v1, a1, T):
    T2 = T**2
    T3 = T**3
    T4 = T**4
    T5 = T**5

    # These 'b' values represent the 'gap' between 
    # the state at T (under constant accel) and the targets.
    # This is standard Hermite interpolation math.
    b_p = p1 - (p0 + v0*T + 0.5*a0*T2)
    b_v = v1 - (v0 + a0*T)
    b_a = a1 - a0

    # The 3x3 matrix inversion results:
    c3 = (10/T3)*b_p  - (4/T2)*b_v  + (0.5/T)*b_a
    c4 = (-15/T4)*b_p + (7/T3)*b_v  - (1/T2)*b_a
    c5 = (6/T5)*b_p   - (3/T4)*b_v  + (0.5/T3)*b_a

    # Note: c2 is a0/2. When evaluating, 
    # use p = c0 + c1*t + c2*t^2 + c3*t^3...
    return [p0, v0, a0/2.0, c3, c4, c5]

def evaluate_spline(coeffs, t):
    """Calculates P, V, A at any time t using Forward Kinematics."""
    c0, c1, c2, c3, c4, c5 = coeffs
    p = c0 + c1*t + c2*t**2 + c3*t**3 + c4*t**4 + c5*t**5
    v = c1 + 2*c2*t + 3*c3*t**2 + 4*c4*t**3 + 5*c5*t**4
    a = 2*c2 + 6*c3*t + 12*c4*t**2 + 20*c5*t**3
    return p, v, a

def move_interp_accel(data):
    # data is given as nx4 vector. format is [px, py, vx, vy]
    segments_per_sec = 20
    steps_per_segment = 10

    dt = 1 / segments_per_sec # overall timestep
    sub_dt = dt / steps_per_segment
    
    p_data = data[:, 0:2]
    v_data = data[:, 2:]

    # preallocate history
    total_sim_steps = (len(p_data) - 3) * steps_per_segment
    pos_history = np.zeros((total_sim_steps + 1, 2))
    motor_history = np.zeros((total_sim_steps, 4))

    # initial state
    curr_p = p_data[0]
    curr_v = v_data[0]
    curr_a = np.array([0.0, 0.0])
    curr_lengths = np.linalg.norm(corner_positions - curr_p, axis=1)
    pos_history[0] = curr_p
    
    idx = 0
    # step the simulation
    for i in range(len(p_data) - 3):
        # do lookahead
        p1, v1 = p_data[i+1], v_data[i+1]
        p2, v2 = p_data[i+2], v_data[i+2]
        p3, v3 = p_data[i+3], v_data[i+3]

        # calculate optimal a1 for both x, y
        a1 = compute_a1(curr_p, p1, p2, p3, curr_v, v1, v2, v3, curr_a)

        # generate spline coefficients
        [c0, c1, c2, c3, c4, c5] = get_quintic_coeffs(curr_p, curr_v, curr_a, p1, v1, a1, dt)

        # inner loop: forward kinematics
        for step in range(0, steps_per_segment):
            t = sub_dt * step

            # evaluate spline at t
            next_p = curr_p + curr_v*t + 0.5*curr_a*t**2 + c3*t**3 + c4*t**4 + c5*t**5

            # inverse kinematics
            next_lengths = np.linalg.norm(corner_positions - next_p, axis=1)
            dl = next_lengths - curr_lengths

            # write results
            pos_history[idx + 1] = next_p
            motor_history[idx] = dl

            # update local loop state
            curr_lengths = next_lengths
            idx += 1

        # set up for next outer loop iter
        curr_p, curr_v, curr_a = next_p, v1, a1
        
    return pos_history, motor_history

def move_interp_accel_test(data):
    segments_per_sec = 20 # 50ms segments
    steps_per_segment = 10 # 5ms simulation ticks
    dt = 1 / segments_per_sec 
    sub_dt = dt / steps_per_segment
    
    p_data = data[:, 0:2]
    v_data = data[:, 2:]

    total_sim_steps = (len(p_data) - 3) * steps_per_segment
    pos_history = np.zeros((total_sim_steps + 1, 2))
    motor_history = np.zeros((total_sim_steps, 4))

    curr_p, curr_v, curr_a = p_data[0], v_data[0], np.array([0.0, 0.0])
    curr_lengths = np.linalg.norm(corner_positions - curr_p, axis=1)
    pos_history[0] = curr_p
    
    idx = 0
    for i in range(len(p_data) - 3):
        p1, v1 = p_data[i+1], v_data[i+1]
        p2, v2 = p_data[i+2], v_data[i+2]
        p3, v3 = p_data[i+3], v_data[i+3]

        a1 = compute_a1(curr_p, p1, p2, p3, curr_v, v1, v2, v3, curr_a)
        [c0, c1, c2, c3, c4, c5] = get_quintic_coeffs(curr_p, curr_v, curr_a, p1, v1, a1, dt)

        # ANCHOR: We keep the starting p, v, a fixed for the inner loop
        start_p, start_v, start_a = curr_p.copy(), curr_v.copy(), curr_a.copy()

        # Step from 1 to 10 so we evaluate the END of each micro-step
        for step in range(1, steps_per_segment + 1):
            t = sub_dt * step

            # Use the ANCHORS (start_p, etc) so t is relative to segment start
            next_p = start_p + start_v*t + 0.5*start_a*t**2 + c3*t**3 + c4*t**4 + c5*t**5

            next_lengths = np.linalg.norm(corner_positions - next_p, axis=1)
            dl = next_lengths - curr_lengths

            pos_history[idx + 1] = next_p
            motor_history[idx] = dl

            curr_lengths = next_lengths
            idx += 1

        # HAND-OFF: Calculate final velocity/accel at t=dt for the next segment
        # This is where we ensure the "green bars" in your plot disappear
        curr_p = next_p # This is now exactly at t=dt
        curr_v = start_v + start_a*dt + 3*c3*dt**2 + 4*c4*dt**3 + 5*c5*dt**4
        curr_a = start_a + 6*c3*dt + 12*c4*dt**2 + 20*c5*dt**3
        
    return pos_history, motor_history

# generates circular trajectory data for testing
def generate_smooth_circle(center=(250, 250), radius=175, duration=5.0, dt=1/20):
    """
    Generates P and V data for a circular path with smooth 
    acceleration and deceleration ramps.
    """
    num_steps = int(duration / dt)
    t = np.linspace(0, 1, num_steps)
    
    # 1. Generate a smooth 's' curve for theta (0 to 2*pi)
    # We use a quintic polynomial ramp: 10t^3 - 15t^4 + 6t^5
    # This ensures velocity and acceleration start/end at 0.
    s = 10*t**3 - 15*t**4 + 6*t**5
    theta = s * (2 * np.pi)
    
    # Derivative of s (for velocity calculation)
    ds_dt = (30*t**2 - 60*t**3 + 30*t**4) / duration
    dtheta_dt = ds_dt * (2 * np.pi)
    
    # 2. Map polar to Cartesian coordinates
    cx, cy = center
    p_x = cx + radius * np.cos(theta)
    p_y = cy + radius * np.sin(theta)
    
    # 3. Calculate Velocities (Chain Rule: dx/dt = dx/dtheta * dtheta/dt)
    v_x = -radius * np.sin(theta) * dtheta_dt
    v_y =  radius * np.cos(theta) * dtheta_dt
    
    # Format into the arrays your controller expects
    p_data = np.vstack((p_x, p_y)).T
    v_data = np.vstack((v_x, v_y)).T
    
    return np.hstack((p_data, v_data))



def plot_static_trajectory(pos_history, motor_history, label=''):
    
    # 1. Create the Figure
    fig, (ax_rob, ax_mot) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- LEFT PLOT: ROBOT WORKSPACE ---
    # Draw the frame and pulleys
    ax_rob.set_aspect('equal')
    ax_rob.grid(True, linestyle=':', alpha=0.6)
    ax_rob.set_title("Robot Workspace (Cartesian Path)")
    ax_rob.set_xlabel("X Position (mm)")
    ax_rob.set_ylabel("Y Position (mm)")
    
    # Plot Pulleys (M1-M4)
    ax_rob.scatter(corner_positions[:,0], corner_positions[:,1], 
                   c='red', marker='s', s=100, label="Pulleys")
    
    # Plot the full path trace
    ax_rob.plot(pos_history[:, 0], pos_history[:, 1], 'b-', linewidth=2, label="Effector Path")
    
    # Plot Start and End points
    ax_rob.scatter(pos_history[0,0], pos_history[0,1], c='green', s=80, label="Start", zorder=5)
    ax_rob.scatter(pos_history[-1,0], pos_history[-1,1], c='purple', s=80, label="End", zorder=5)
    
    # Draw Cables at the END position for context
    end_pos = pos_history[-1]
    for i in range(4):
        px, py = corner_positions[i]
        ax_rob.plot([px, end_pos[0]], [py, end_pos[1]], 'k-', linewidth=1, alpha=0.3)
    
    ax_rob.legend(loc='upper right', fontsize='small')

    # --- RIGHT PLOT: MOTOR DELTA L ---
    ax_mot.set_title("Motor Step Increments ($\Delta L$)")
    ax_mot.set_xlabel("Tick Number")
    ax_mot.set_ylabel("Delta L (mm per tick)")
    ax_mot.grid(True, alpha=0.3)
    
    # Plot a line for each motor's velocity/delta history
    ticks = np.arange(len(motor_history))
    colors = ['#1f77b4', "#fe8307", '#2ca02c', '#d62728'] # Standard distinct colors
    
    ax_mot.plot(ticks, motor_history[:, 0], label="Bottom Left Motor", color=colors[0], linewidth=2)
    ax_mot.plot(ticks, motor_history[:, 1], label="Bottom Right Motor", color=colors[1], linewidth=2)
    ax_mot.plot(ticks, motor_history[:, 2], label="Top Left Motor", color=colors[2], linewidth=2)
    ax_mot.plot(ticks, motor_history[:, 3], label="Top Right Motor", color=colors[3], linewidth=2)    

    ax_mot.axhline(0, color='black', linewidth=1, alpha=0.5) # Zero line for direction reference
    ax_mot.legend(loc='best', fontsize='small')

    if label:
        plt.title("Cable Robot trajectory simulation: " + label)
    else:
        plt.title("Cable Robot trajectory simulation")

    plt.legend()
    plt.tight_layout()
    plt.show()

def animate_trajectory(pos_history, motor_history, time, label=''):
    # compute velocities from pos_history
    ticks = pos_history.shape[0]
    ticks_per_sec = int(ticks / time)
    cartesian_vels = np.linalg.norm(np.diff(pos_history, axis=0), axis=1) * ticks_per_sec
    
    fig, (ax_rob, ax_mot, ax_vel) = plt.subplots(1, 3, figsize=(21, 7))
    
    # --- LEFT PLOT: ROBOT MOTION ---
    ax_rob.set_xlim(-50, 550) 
    ax_rob.set_ylim(-50, 550)
    ax_rob.set_aspect('equal')
    ax_rob.grid(True, linestyle=':', alpha=0.6)
    ax_rob.set_title("Robot Workspace")
    
    # Draw Pulleys
    ax_rob.scatter(corner_positions[:,0], corner_positions[:,1], c='red', marker='s', s=100)
    
    effector, = ax_rob.plot([], [], 'bo', markersize=8)
    cables = [ax_rob.plot([], [], 'k-', linewidth=1, alpha=0.4)[0] for _ in range(4)]
    trace, = ax_rob.plot([], [], 'b--', linewidth=1, alpha=0.3)

    # --- MIDDLE PLOT: motor history ---
    ax_mot.set_xlim(0, len(motor_history))
    
    # Center Y-axis around zero
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

    # --- RIGHT PLOT: CARTESIAN VELOCITY ---
    ax_vel.set_xlim(0, len(pos_history))
    v_max = np.max(cartesian_vels) * 1.2 if len(cartesian_vels) > 0 else 1.0
    ax_vel.set_ylim(0, v_max)
    ax_vel.set_title("End-Effector Velocity")
    ax_vel.set_ylabel("Velocity (mm/s)")
    ax_vel.set_xlabel("Tick Number")
    ax_vel.grid(True, alpha=0.3)

    vel_line, = ax_vel.plot([], [], 'g-', linewidth=2, label="Magnitude")
    ax_vel.legend(loc='upper right', fontsize='x-small')

    def init():
        effector.set_data([], [])
        trace.set_data([], [])
        for line in cables + mot_lines:
            line.set_data([], [])
        return [effector, trace] + cables + mot_lines

    def update(frame):
        # Update Workspace
        curr_x, curr_y = pos_history[frame]
        effector.set_data([curr_x], [curr_y])
        trace.set_data(pos_history[:frame, 0], pos_history[:frame, 1])
        
        for i in range(4):
            px, py = corner_positions[i]
            cables[i].set_data([px, curr_x], [py, curr_y])
            
        # Update Motor Plot
        if frame > 0:
            ticks = np.arange(frame)
            for i in range(4):
                mot_lines[i].set_data(ticks, motor_history[:frame, i])

        # Update Velocity Plot
            vel_line.set_data(np.arange(frame), cartesian_vels[:frame])
        
        return [effector, trace, vel_line] + cables + mot_lines

    ani = FuncAnimation(fig, update, frames=len(pos_history),
                        init_func=init, blit=True, interval=2)
    # interval = time/ticks for real-time plotting

    plt.legend()
    plt.show()

# startx = 50
# starty = 150
# endx = 400
# endy = 475

# start_pos = np.array([startx, starty], dtype=float)
# end_pos = np.array([endx, endy], dtype=float)
# time = 0.25

# # pos_test, motor_test = move_interp_noaccel(start_pos, end_pos, 1)

# pos_test, motor_test = move_interp_accel(start_pos, end_pos, time)

# print(motor_test)


circle_data = generate_smooth_circle()
pos_test, motor_test = move_interp_accel_test(circle_data)

np.set_printoptions(threshold=np.inf, precision=6, suppress=True)
print(pos_test)

time = len(motor_test) / 1000


# print(pos_test)
animate_trajectory(pos_test, motor_test, time, label='interpolated traj')
# plot_static_trajectory(pos_test, motor_test, label='interpolated traj')