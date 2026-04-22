import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from config import CORNERS

LOG = """
Move: [-223.4  -10.3] → [-200.   80.]  (1.00s)
Done. Actual: [-203.3   66.7]  Error: 13.70 mm
Move: [-203.2   66.9] → [-200.  -80.]  (1.00s)
Done. Actual: [-219.8  -67.4]  Error: 23.48 mm
Move: [-219.9  -67.5] → [-300.    0.]  (1.00s)
Done. Actual: [-284.3   -9.2]  Error: 18.19 mm
""".strip()

def parse_log(text):
    moves = []
    lines = text.strip().split('\n')
    for i in range(0, len(lines), 2):
        # Parse "Move: [x y] → [x y]  (Ns)"
        nums = re.findall(r'[-+]?\d*\.?\d+', lines[i])
        start = [float(nums[0]), float(nums[1])]
        target = [float(nums[2]), float(nums[3])]
        # Parse "Done. Actual: [x y]  Error: N mm"
        nums2 = re.findall(r'[-+]?\d*\.?\d+', lines[i+1])
        actual = [float(nums2[0]), float(nums2[1])]
        moves.append({'start': start, 'target': target, 'actual': actual})
    return moves

moves = parse_log(LOG)

fig, ax = plt.subplots(figsize=(10, 8))

# Draw workspace (convex hull of corners)
hull = np.vstack([CORNERS, CORNERS[0]])
ax.plot(hull[:, 0], hull[:, 1], 'k-', linewidth=2, label='Workspace')
for i, c in enumerate(CORNERS):
    ax.plot(c[0], c[1], 'ks', markersize=10)
    ax.annotate(f'M{i+1}', c, textcoords="offset points", xytext=(8, 8), fontsize=11)

# Draw each move
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
for i, m in enumerate(moves):
    s = np.array(m['start'])
    t = np.array(m['target'])
    a = np.array(m['actual'])

    # Planned path (dashed)
    ax.annotate('', xy=t, xytext=s,
                arrowprops=dict(arrowstyle='->', color=colors[i % len(colors)], linestyle='--', lw=1.5))

    # Actual path (solid)
    ax.annotate('', xy=a, xytext=s,
                arrowprops=dict(arrowstyle='->', color=colors[i % len(colors)], linestyle='-', lw=2))

    ax.plot(*s, 'o', color=colors[i % len(colors)], markersize=8)
    ax.plot(*t, 'x', color=colors[i % len(colors)], markersize=12, markeredgewidth=3)
    ax.plot(*a, 'd', color=colors[i % len(colors)], markersize=8)

    err = np.linalg.norm(a - t)
    ax.annotate(f'Move {i+1} err={err:.0f}mm', xy=a, textcoords="offset points",
                xytext=(10, -15), fontsize=9, color=colors[i % len(colors)])

# Legend
ax.plot([], [], 'k--', label='Planned path')
ax.plot([], [], 'k-', label='Actual path')
ax.plot([], [], 'kx', markersize=10, markeredgewidth=2, label='Target')
ax.plot([], [], 'kd', markersize=7, label='Actual end')
ax.plot([], [], 'ko', markersize=7, label='Start')

ax.legend(loc='lower left')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_title('Cable Robot Moves: Planned vs Actual')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('move_visualization.png', dpi=150)
plt.show()
print("Saved to move_visualization.png")
