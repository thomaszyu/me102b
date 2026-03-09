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

def move_naive(start, end, time):
    # move at constant MOTOR rate
    # start, end are 2-vectors with start and end positions
    ticks_per_sec = 100 # i guess, why not
    total_ticks = int(time*ticks_per_sec)

    lengths_start = np.linalg.norm(corner_positions - start, axis=1)
    lengths_end = np.linalg.norm(corner_positions - end, axis=1)
    dl = (lengths_end - lengths_start) / total_ticks

    curr_lengths = lengths_start.copy()
    pos_history = np.zeros((total_ticks + 1, 2))
    pos_history[0] = start

    for i in range(total_ticks):
        curr_lengths += dl
        curr_pos = forward_kinematics(curr_lengths)
        pos_history[i+1] = curr_pos

    return pos_history

def move_interp_noaccel(start, end, time):
    # move at constant LINEAR rate (with interpolation)
    # start, end are 2-vectors with start and end positions
    ticks_per_sec = 100 # i guess, why not
    total_ticks = int(time*ticks_per_sec)

    vector = end - start
    dx = vector / total_ticks # ideal linear motion per timestep

    pos_history = np.zeros((total_ticks + 1, 2))
    pos_history[0] = start
    curr_pos = start.copy()
    curr_lengths = np.linalg.norm(corner_positions - curr_pos, axis=1)

    motor_history = np.zeros((total_ticks, 4))

    for i in range(total_ticks):
        # calculate required next lengths
        next_lengths = np.linalg.norm(corner_positions - (curr_pos + dx), axis=1)
        dl = next_lengths - curr_lengths

        curr_lengths = next_lengths 

        # within the timestep, move at constant rate
        curr_pos += dx
        pos_history[i+1] = curr_pos
        motor_history[i] = dl

    return pos_history, motor_history

def move_interp_accel(start, end, time):
    # Higher tick rate recommended for high-speed quintic paths
    ticks_per_sec = 200 
    total_ticks = int(time * ticks_per_sec)
    
    pos_history = np.zeros((total_ticks + 1, 2))
    motor_history = np.zeros((total_ticks, 4))

    curr_pos = start.copy()
    curr_lengths = np.linalg.norm(corner_positions - curr_pos, axis=1)
    pos_history[0] = curr_pos
    
    for i in range(total_ticks):
        # normalized time t goes from 0.0 to 1.0
        t = i / total_ticks
        
        # quintic Polynomial: s(t) = 10t^3 - 15t^4 + 6t^5
        # this profile has zero velocity AND zero acceleration at t=0 and t=1
        s = 10*t**3 - 15*t**4 + 6*t**5 
        # the equation comes from solving s(1) = 1 
        # and s(0) = s'(0) = s''(0) = s'(1) = s''(1) = 0
        # 6 dof equation, hence 5th degree polynomial is needed
        
        # 1. Calculate Cartesian Positions
        next_pos = start + s * (end - start)
        
        # 2. Inverse Kinematics: Calculate Absolute Cable Lengths
        next_lengths = np.linalg.norm(corner_positions - next_pos, axis=1)
        dl = next_lengths - curr_lengths

        # 3. Store Results
        pos_history[i+1] = next_pos
        motor_history[i] = dl

        # 4. Set up for next loop
        curr_pos = next_pos
        curr_lengths = next_lengths
        
    return pos_history, motor_history

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
                        init_func=init, blit=True, interval=20)
    # interval = time/ticks for real-time plotting

    plt.legend()
    plt.show()

startx = 50
starty = 150
endx = 400
endy = 475

start_pos = np.array([startx, starty], dtype=float)
end_pos = np.array([endx, endy], dtype=float)
time = 0.25

# pos_test, motor_test = move_interp_noaccel(start_pos, end_pos, 1)

pos_test, motor_test = move_interp_accel(start_pos, end_pos, time)

# print(motor_test)

# print(pos_test)
animate_trajectory(pos_test, motor_test, time, label='interpolated traj')
# plot_static_trajectory(pos_test, motor_test, label='interpolated traj')