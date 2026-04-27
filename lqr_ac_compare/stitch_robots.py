"""
Stitch gen_robot_1 through gen_robot_5 into a 5-second GIF with fade transitions.
"""

import numpy as np
from PIL import Image
import os

# Parameters
N_IMAGES = 5
TOTAL_DURATION = 5.0  # seconds
FPS = 30
FADE_DURATION = 0.5  # seconds for fade transition

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load images and get target size from first image
images_raw = []
for i in range(1, N_IMAGES + 1):
    path = os.path.join(script_dir, f"gen_robot_{i}.png")
    img = Image.open(path).convert("RGBA")
    images_raw.append(img)
    print(f"Loaded gen_robot_{i}.png: {img.size}")

# Use first image dimensions as target
target_size = images_raw[0].size  # (width, height)
print(f"Target dimensions: {target_size[0]}x{target_size[1]}")

# Resize all images to match first image
images = []
for i, img in enumerate(images_raw):
    if img.size != target_size:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        print(f"  Resized gen_robot_{i+1}.png to {target_size}")
    images.append(np.array(img))

height, width = images[0].shape[:2]

# Calculate timing
time_per_image = TOTAL_DURATION / N_IMAGES  # 1 second each
fade_frames = int(FADE_DURATION * FPS)
total_frames = int(TOTAL_DURATION * FPS)

print(f"Total frames: {total_frames}, Fade frames: {fade_frames}")

# Generate frames
frames = []
for frame_idx in range(total_frames):
    t = frame_idx / FPS  # current time in seconds

    # Determine which image we're on
    img_idx = min(int(t / time_per_image), N_IMAGES - 1)
    time_in_image = t - img_idx * time_per_image

    # Check if we're in a fade transition
    if time_in_image > (time_per_image - FADE_DURATION) and img_idx < N_IMAGES - 1:
        # Fading from img_idx to img_idx + 1
        fade_progress = (time_in_image - (time_per_image - FADE_DURATION)) / FADE_DURATION
        fade_progress = min(1.0, max(0.0, fade_progress))

        # Blend images
        img1 = images[img_idx].astype(float)
        img2 = images[img_idx + 1].astype(float)
        blended = (1 - fade_progress) * img1 + fade_progress * img2
        frame = blended.astype(np.uint8)
    else:
        frame = images[img_idx]

    # Convert to PIL Image (RGB for GIF)
    frame_rgb = Image.fromarray(frame).convert("RGB")
    frames.append(frame_rgb)

# Save GIF
output_path = os.path.join(script_dir, "gen_robots_combined.gif")
frames[0].save(
    output_path,
    save_all=True,
    append_images=frames[1:],
    duration=int(1000 / FPS),  # milliseconds per frame
    loop=0
)

print(f"Saved: {output_path}")
print(f"Duration: {TOTAL_DURATION}s, {len(frames)} frames at {FPS} FPS")
