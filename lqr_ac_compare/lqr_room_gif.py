"""
LQR trajectory visualization in a room environment.

Simulates a robot navigating to a target using LQR control with stochastic noise.
Uses Euler-Maruyama discretization for the SDE:
    dx = -K(x - x_target) dt + sigma dW
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image
import os

# Room and problem setup
ROOM_WIDTH = 1530
ROOM_HEIGHT = 770
START_POS = np.array([229.0, 590.0])
TARGET_POS = np.array([1329.0, 137.0])

# Simulation parameters
T_FINAL = 6.0  # seconds
DT = 0.01  # time step
N_STEPS = int(T_FINAL / DT)
SIGMA = 1.0  # noise coefficient
N_PATHS = 5

# LQR gain (tuned to reach target in ~10 seconds)
# For dx/dt = -K(x - target), solution is x(t) = target + (x0 - target) * exp(-K*t)
# With K=0.4, after 10 seconds: exp(-4) ≈ 0.018, so ~98% of the way there
K_GAIN = 0.4


def simulate_path(start: np.ndarray, target: np.ndarray, seed: int) -> np.ndarray:
    """
    Simulate one sample path using Euler-Maruyama.

    SDE: dx = -K(x - target) dt + sigma * dW

    Returns:
        trajectory: (N_STEPS+1, 2) array of positions
    """
    rng = np.random.default_rng(seed)

    trajectory = np.zeros((N_STEPS + 1, 2))
    trajectory[0] = start.copy()

    x = start.copy()
    for i in range(N_STEPS):
        # Control: head straight to target
        u = -K_GAIN * (x - target)

        # Euler-Maruyama step
        dW = rng.standard_normal(2) * np.sqrt(DT)
        x = x + u * DT + SIGMA * dW * 100  # scale noise to room coordinates

        # Clip to room bounds
        x[0] = np.clip(x[0], 0, ROOM_WIDTH)
        x[1] = np.clip(x[1], 0, ROOM_HEIGHT)

        trajectory[i + 1] = x

    return trajectory


def create_gif(
    trajectories: list,
    background_path: str,
    output_path: str,
    fps: int = 30,
):
    """
    Create animated GIF of trajectories on background image.

    Args:
        trajectories: list of (N_STEPS+1, 2) arrays
        background_path: path to background image
        output_path: path for output GIF
        fps: frames per second
    """
    # Load background
    bg_img = Image.open(background_path)
    bg_array = np.array(bg_img)

    # Colors for different paths
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    # Set up figure
    fig, ax = plt.subplots(figsize=(15.3, 7.7), dpi=100)
    ax.imshow(bg_array, extent=[0, ROOM_WIDTH, ROOM_HEIGHT, 0])
    ax.set_xlim(0, ROOM_WIDTH)
    ax.set_ylim(ROOM_HEIGHT, 0)  # Flip y-axis (image coordinates)
    ax.set_aspect('equal')
    ax.axis('off')

    # Initialize line objects for each path
    lines = []
    for i in range(N_PATHS):
        line, = ax.plot([], [], color=colors[i], linewidth=5, alpha=0.8)
        lines.append(line)


    # Frame skip for smoother animation (don't render every simulation step)
    frame_skip = max(1, N_STEPS // (fps * int(T_FINAL)))
    n_frames = N_STEPS // frame_skip + 1

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(frame):
        idx = min(frame * frame_skip, N_STEPS)

        for i, (line, traj) in enumerate(zip(lines, trajectories)):
            line.set_data(traj[:idx+1, 0], traj[:idx+1, 1])

        return lines

    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=n_frames, interval=1000/fps, blit=True
    )

    # Save GIF
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Saved: {output_path}")


def create_static_plot(trajectories: list, background_path: str, output_path: str):
    """Create static plot showing all trajectories."""
    bg_img = Image.open(background_path)
    bg_array = np.array(bg_img)

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    fig, ax = plt.subplots(figsize=(15.3, 7.7), dpi=100)
    ax.imshow(bg_array, extent=[0, ROOM_WIDTH, ROOM_HEIGHT, 0])

    for i, traj in enumerate(trajectories):
        ax.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=5,
                alpha=0.8, label=f'Path {i+1}')

    ax.set_xlim(0, ROOM_WIDTH)
    ax.set_ylim(ROOM_HEIGHT, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title(f'LQR Trajectories (K={K_GAIN}, σ={SIGMA})', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    background_path = os.path.join(script_dir, "base_env.png")

    if not os.path.exists(background_path):
        raise FileNotFoundError(f"Background image not found: {background_path}")

    print(f"Room: {ROOM_WIDTH} x {ROOM_HEIGHT}")
    print(f"Start: {START_POS}")
    print(f"Target: {TARGET_POS}")
    print(f"Distance: {np.linalg.norm(TARGET_POS - START_POS):.1f} pixels")
    print(f"Simulation: T={T_FINAL}s, dt={DT}, steps={N_STEPS}")
    print(f"Control gain K={K_GAIN}, noise σ={SIGMA}")
    print()

    # Simulate paths
    print("Simulating paths...")
    trajectories = []
    for i in range(N_PATHS):
        traj = simulate_path(START_POS, TARGET_POS, seed=42 + i)
        trajectories.append(traj)
        final_dist = np.linalg.norm(traj[-1] - TARGET_POS)
        print(f"  Path {i+1}: final distance to target = {final_dist:.1f} pixels")

    # Create outputs
    print("\nCreating visualizations...")

    # Static plot
    static_path = os.path.join(script_dir, "lqr_trajectories.png")
    create_static_plot(trajectories, background_path, static_path)

    # Animated GIF
    gif_path = os.path.join(script_dir, "lqr_trajectories.gif")
    create_gif(trajectories, background_path, gif_path, fps=30)

    print("\nDone!")


if __name__ == "__main__":
    main()
